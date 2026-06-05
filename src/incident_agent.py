from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .knowledge_db import initialize_database
from .slm_client import SlmLoadBalancer, parse_slm_json


AGENT_MODEL = "google/gemma-3-4b"


def generate_incident_report(
    db_path: str | Path,
    instance: str | None = None,
    *,
    log_entry_id: int | None = None,
    agent_model: str = AGENT_MODEL,
    base_url: str = "http://127.0.0.1:1234/v1",
) -> dict[str, object]:
    initialize_database(db_path)
    evidence = load_incident_evidence(db_path, instance, log_entry_id=log_entry_id)
    if not evidence:
        raise ValueError(f"No evidence found for instance={instance} log_entry_id={log_entry_id}")

    balancer = SlmLoadBalancer(base_url=base_url, models=[agent_model], timeout_seconds=120)
    raw = ""
    try:
        response = balancer.summarize_observation(_incident_prompt_evidence(evidence))
        raw = response.content
        parsed = parse_slm_json(raw)
    except Exception as exc:
        raw = f"Agent fallback used: {exc}"
        parsed = _fallback_report(evidence)

    report = {
        "run_id": evidence.get("run_id"),
        "log_entry_id": evidence.get("log_entry", {}).get("id"),
        "instance": evidence["instance"],
        "model": evidence.get("best_model", ""),
        "severity": str(parsed.get("severity", parsed.get("risk", evidence.get("severity", "warning")))),
        "title": str(parsed.get("title", _fallback_report(evidence)["title"])),
        "summary": str(parsed.get("summary", "")),
        "likely_root_cause": str(parsed.get("likely_root_cause", _likely_root_cause(evidence))),
        "recommendations": json.dumps(_recommendations_from_parsed(parsed), default=str),
        "evidence_json": json.dumps(evidence, default=str),
        "agent_model": agent_model,
        "agent_raw": raw,
    }
    report["id"] = insert_incident_report(db_path, report)
    return report


def load_incident_evidence(
    db_path: str | Path,
    instance: str | None = None,
    *,
    log_entry_id: int | None = None,
) -> dict[str, object]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        log_entry = None
        if log_entry_id is not None:
            log_entry = conn.execute("SELECT * FROM log_entries WHERE id = ?", (log_entry_id,)).fetchone()
            if log_entry is None:
                return {}
            instance = str(log_entry["instance"])
        if not instance:
            return {}
        run = conn.execute("SELECT id FROM model_runs ORDER BY id DESC LIMIT 1").fetchone()
        run_id = int(run["id"]) if run else None
        model = conn.execute(
            """
            SELECT model FROM model_scores
            WHERE run_id = ?
            ORDER BY host_mean_score_auc DESC, f1 DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        best_model = str(model["model"]) if model else ""
        host_score = conn.execute(
            "SELECT * FROM host_scores WHERE run_id = ? AND instance = ? AND model = ?",
            (run_id, instance, best_model),
        ).fetchone()
        observation = None
        if log_entry_id is not None:
            observation = conn.execute(
                """
                SELECT * FROM observations
                WHERE run_id = ? AND log_entry_id = ?
                ORDER BY confidence DESC LIMIT 1
                """,
                (run_id, log_entry_id),
            ).fetchone()
        if observation is None:
            observation = conn.execute(
                """
                SELECT * FROM observations
                WHERE run_id = ? AND instance = ? AND log_entry_id IS NULL
                ORDER BY confidence DESC LIMIT 1
                """,
                (run_id, instance),
            ).fetchone()
        alerts = conn.execute(
            "SELECT * FROM alert_windows WHERE instance = ? ORDER BY alertname, mountpoint",
            (instance,),
        ).fetchall()
        feedback = conn.execute(
            """
            SELECT feedback_type, note, user, created_at FROM feedback
            WHERE instance = ? OR (log_entry_id IS NOT NULL AND log_entry_id = ?)
            ORDER BY id DESC LIMIT 10
            """,
            (instance, log_entry_id),
        ).fetchall()
        entry_slm_extraction = None
        entry_model_output = None
        if log_entry_id is not None:
            entry_slm_extraction = conn.execute(
                "SELECT * FROM entry_slm_extractions WHERE log_entry_id = ? ORDER BY id DESC LIMIT 1",
                (log_entry_id,),
            ).fetchone()
            entry_model_output = conn.execute(
                "SELECT * FROM entry_model_outputs WHERE log_entry_id = ? ORDER BY id DESC LIMIT 1",
                (log_entry_id,),
            ).fetchone()
    finally:
        conn.close()

    if host_score is None and observation is None and not alerts:
        return {}
    evidence_json = {}
    if observation and observation["evidence_json"]:
        try:
            evidence_json = json.loads(observation["evidence_json"])
        except json.JSONDecodeError:
            evidence_json = {"raw": observation["evidence_json"]}
    return {
        "run_id": run_id,
        "instance": instance,
        "log_entry": dict(log_entry) if log_entry else {},
        "best_model": best_model,
        "severity": observation["severity"] if observation else ("critical" if alerts else "warning"),
        "host_score": dict(host_score) if host_score else {},
        "observation": dict(observation) if observation else {},
        "entry_slm_extraction": dict(entry_slm_extraction) if entry_slm_extraction else {},
        "entry_model_output": dict(entry_model_output) if entry_model_output else {},
        "observation_evidence": evidence_json,
        "alerts": [dict(row) for row in alerts],
        "recent_feedback": [dict(row) for row in feedback],
    }


def insert_incident_report(db_path: str | Path, report: dict[str, object]) -> int:
    initialize_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO incident_reports(
                run_id, log_entry_id, instance, model, severity, title, summary, likely_root_cause,
                recommendations, evidence_json, agent_model, agent_raw
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.get("run_id"),
                report.get("log_entry_id"),
                report["instance"],
                report.get("model", ""),
                report.get("severity", ""),
                report.get("title", ""),
                report.get("summary", ""),
                report.get("likely_root_cause", ""),
                report.get("recommendations", "[]"),
                report.get("evidence_json", "{}"),
                report.get("agent_model", ""),
                report.get("agent_raw", ""),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def _incident_prompt_evidence(evidence: dict[str, object]) -> dict[str, object]:
    return {
        "task": (
            "Create an incident report from CIDT anomaly evidence. Return compact JSON only with keys "
            "title, severity, summary, likely_root_cause, recommendations, and confidence_reason. "
            "recommendations must be a list of short strings."
        ),
        "evidence": evidence,
    }


def _fallback_report(evidence: dict[str, object]) -> dict[str, object]:
    observation = evidence.get("observation", {})
    alerts = evidence.get("alerts", [])
    alertnames = sorted({alert.get("alertname", "") for alert in alerts if alert.get("alertname")})
    summary = observation.get("summary") or f"{evidence['instance']} has anomaly evidence from model scores and alerts."
    recommendation = observation.get("recommendation") or "Review telemetry drivers, active alerts, and recent changes."
    return {
        "title": f"{'; '.join(alertnames) or 'Anomaly'} on {evidence['instance']}",
        "severity": observation.get("severity", "warning"),
        "summary": summary,
        "likely_root_cause": "Likely related to the active alert context and top metric deviations.",
        "recommendations": [recommendation],
        "confidence_reason": "Fallback report generated from structured observations and alert windows.",
    }


def _recommendations_from_parsed(parsed: dict[str, object]) -> list[str]:
    recommendations = parsed.get("recommendations", [])
    if isinstance(recommendations, list):
        if recommendations:
            return [str(item) for item in recommendations]
    if recommendations:
        return [str(recommendations)]
    recommendation = parsed.get("recommendation", "")
    return [str(recommendation)] if recommendation else []


def _likely_root_cause(evidence: dict[str, object]) -> str:
    alerts = evidence.get("alerts", [])
    alertnames = {alert.get("alertname", "") for alert in alerts}
    if "HostOutOfDiskSpace" in alertnames:
        return "Sustained filesystem capacity pressure on the affected mountpoint or backing device."
    if "HostFilesystemDeviceError" in alertnames:
        return "Filesystem or mounted storage device error, likely involving the recorded device or mountpoint."
    return "Likely related to active alert context and anomalous telemetry drivers."
