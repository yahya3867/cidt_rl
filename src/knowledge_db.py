from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .slm_client import parse_slm_json

if TYPE_CHECKING:
    from .slm_client import SlmLoadBalancer


SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    instance TEXT PRIMARY KEY,
    ip TEXT,
    rack TEXT,
    unit TEXT,
    note TEXT,
    assigned_user TEXT
);

CREATE TABLE IF NOT EXISTS alert_windows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    instance TEXT NOT NULL,
    alertname TEXT NOT NULL,
    severity TEXT,
    start_time TEXT,
    end_time TEXT,
    device TEXT,
    mountpoint TEXT,
    sample_count INTEGER
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    results_dir TEXT NOT NULL,
    metrics_path TEXT,
    alert_windows_path TEXT,
    summary_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    precision REAL,
    recall REAL,
    f1 REAL,
    balanced_accuracy REAL,
    host_mean_score_auc REAL,
    host_average_precision REAL,
    alert_host_recall REAL,
    healthy_host_false_positive_rate REAL,
    FOREIGN KEY(run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS host_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    instance TEXT NOT NULL,
    alert_host INTEGER,
    max_likelihood REAL,
    mean_likelihood REAL,
    max_anomaly_score REAL,
    mean_anomaly_score REAL,
    predicted_points INTEGER,
    test_points INTEGER,
    FOREIGN KEY(run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    log_entry_id INTEGER,
    instance TEXT NOT NULL,
    observation_type TEXT NOT NULL,
    severity TEXT,
    confidence REAL,
    summary TEXT,
    evidence_json TEXT,
    recommendation TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id),
    FOREIGN KEY(log_entry_id) REFERENCES log_entries(id)
);

CREATE TABLE IF NOT EXISTS log_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    entry_type TEXT NOT NULL,
    timestamp TEXT,
    instance TEXT,
    severity TEXT,
    source TEXT,
    message TEXT,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS incident_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    log_entry_id INTEGER,
    instance TEXT NOT NULL,
    model TEXT,
    severity TEXT,
    title TEXT,
    summary TEXT,
    likely_root_cause TEXT,
    recommendations TEXT,
    evidence_json TEXT,
    agent_model TEXT,
    agent_raw TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id),
    FOREIGN KEY(log_entry_id) REFERENCES log_entries(id)
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_entry_id INTEGER,
    instance TEXT NOT NULL,
    model TEXT,
    feedback_type TEXT NOT NULL,
    note TEXT,
    user TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(log_entry_id) REFERENCES log_entries(id)
);

CREATE TABLE IF NOT EXISTS telemetry_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    timestamp TEXT,
    instance TEXT,
    up REAL,
    cpu_usage_percent REAL,
    mem_used_percent REAL,
    disk_used_percent REAL,
    load1 REAL,
    load5 REAL,
    load15 REAL,
    payload_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id)
);

CREATE TABLE IF NOT EXISTS entry_model_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    log_entry_id INTEGER NOT NULL,
    instance TEXT,
    model TEXT,
    output_type TEXT,
    alert_host INTEGER,
    max_likelihood REAL,
    mean_likelihood REAL,
    max_anomaly_score REAL,
    mean_anomaly_score REAL,
    predicted_points INTEGER,
    test_points INTEGER,
    evidence_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id),
    FOREIGN KEY(log_entry_id) REFERENCES log_entries(id)
);

CREATE TABLE IF NOT EXISTS entry_slm_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    log_entry_id INTEGER NOT NULL,
    instance TEXT,
    model TEXT,
    summary TEXT,
    signal_type TEXT,
    risk TEXT,
    recommendation TEXT,
    confidence_reason TEXT,
    entities_json TEXT,
    extraction_json TEXT,
    raw_response TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(run_id) REFERENCES model_runs(id),
    FOREIGN KEY(log_entry_id) REFERENCES log_entries(id)
);
"""


def initialize_database(db_path: str | Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _ensure_column(conn, "observations", "log_entry_id", "INTEGER")
        conn.commit()
    finally:
        conn.close()


def ingest_alert_anomaly_results(
    db_path: str | Path,
    results_dir: str | Path,
    *,
    metrics_path: str | Path,
    alert_windows_path: str | Path,
    observations: list[dict[str, object]] | None = None,
    slm_balancer: "SlmLoadBalancer | None" = None,
) -> int:
    db_path = Path(db_path)
    results_dir = Path(results_dir)
    initialize_database(db_path)

    summary_path = results_dir / "summary.json"
    model_scores = pd.read_csv(results_dir / "model_scores.csv")
    host_scores = pd.read_csv(results_dir / "host_scores.csv")
    alert_windows = pd.read_csv(results_dir / "alert_windows.csv")
    hosts = _host_rows(metrics_path)
    summary_json = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "{}"

    conn = sqlite3.connect(db_path)
    try:
        run_id = _insert_model_run(conn, results_dir, metrics_path, alert_windows_path, summary_json)
        _replace_hosts(conn, hosts)
        _append_alert_windows(conn, alert_windows)
        _append_telemetry_points(conn, run_id, metrics_path)
        _append_model_scores(conn, run_id, model_scores)
        _append_host_scores(conn, run_id, host_scores)
        if observations:
            _append_observations(conn, run_id, observations)
        _append_log_entries(conn, run_id, alert_windows, host_scores, observations or [], slm_balancer=slm_balancer)
        conn.commit()
    finally:
        conn.close()
    return run_id


def _insert_model_run(
    conn: sqlite3.Connection,
    results_dir: Path,
    metrics_path: str | Path,
    alert_windows_path: str | Path,
    summary_json: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO model_runs(results_dir, metrics_path, alert_windows_path, summary_json) VALUES (?, ?, ?, ?)",
        (str(results_dir), str(metrics_path), str(alert_windows_path), summary_json),
    )
    return int(cursor.lastrowid)


def _host_rows(metrics_path: str | Path) -> pd.DataFrame:
    columns = ["instance", "ip", "rack", "unit", "note", "assigned_user"]
    df = pd.read_csv(metrics_path, usecols=columns)
    return df.drop_duplicates(subset=["instance"])


def _replace_hosts(conn: sqlite3.Connection, hosts: pd.DataFrame) -> None:
    rows = [
        (
            row["instance"],
            row.get("ip", ""),
            row.get("rack", ""),
            str(row.get("unit", "")),
            row.get("note", ""),
            row.get("assigned_user", ""),
        )
        for row in hosts.fillna("").to_dict("records")
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO hosts(instance, ip, rack, unit, note, assigned_user) VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _append_telemetry_points(conn: sqlite3.Connection, run_id: int, metrics_path: str | Path) -> None:
    fields = [
        "timestamp",
        "instance",
        "up",
        "cpu_usage_percent",
        "mem_used_percent",
        "disk_used_percent",
        "load1",
        "load5",
        "load15",
    ]
    for chunk in pd.read_csv(metrics_path, chunksize=10_000):
        chunk = chunk.fillna("")
        rows = []
        for row in chunk.to_dict("records"):
            rows.append(
                (
                    run_id,
                    str(row.get("timestamp", "")),
                    str(row.get("instance", "")),
                    _float_or_none(row.get("up")),
                    _float_or_none(row.get("cpu_usage_percent")),
                    _float_or_none(row.get("mem_used_percent")),
                    _float_or_none(row.get("disk_used_percent")),
                    _float_or_none(row.get("load1")),
                    _float_or_none(row.get("load5")),
                    _float_or_none(row.get("load15")),
                    json.dumps(row, default=str),
                )
            )
        conn.executemany(
            """
            INSERT INTO telemetry_points(
                run_id, timestamp, instance, up, cpu_usage_percent, mem_used_percent,
                disk_used_percent, load1, load5, load15, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _append_alert_windows(conn: sqlite3.Connection, alert_windows: pd.DataFrame) -> None:
    rows = [
        (
            row["instance"],
            row["alertname"],
            row.get("severity", ""),
            row.get("start_time", ""),
            row.get("end_time", ""),
            row.get("device", ""),
            row.get("mountpoint", ""),
            int(row.get("sample_count", 0) or 0),
        )
        for row in alert_windows.fillna("").to_dict("records")
    ]
    conn.executemany(
        """
        INSERT INTO alert_windows(instance, alertname, severity, start_time, end_time, device, mountpoint, sample_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _append_model_scores(conn: sqlite3.Connection, run_id: int, model_scores: pd.DataFrame) -> None:
    fields = [
        "model",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "host_mean_score_auc",
        "host_average_precision",
        "alert_host_recall",
        "healthy_host_false_positive_rate",
    ]
    rows = [(run_id, *[row.get(field, None) for field in fields]) for row in model_scores.to_dict("records")]
    conn.executemany(
        """
        INSERT INTO model_scores(
            run_id, model, precision, recall, f1, balanced_accuracy, host_mean_score_auc,
            host_average_precision, alert_host_recall, healthy_host_false_positive_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _append_host_scores(conn: sqlite3.Connection, run_id: int, host_scores: pd.DataFrame) -> None:
    fields = [
        "model",
        "instance",
        "alert_host",
        "max_likelihood",
        "mean_likelihood",
        "max_anomaly_score",
        "mean_anomaly_score",
        "predicted_points",
        "test_points",
    ]
    rows = [(run_id, *[row.get(field, None) for field in fields]) for row in host_scores.to_dict("records")]
    conn.executemany(
        """
        INSERT INTO host_scores(
            run_id, model, instance, alert_host, max_likelihood, mean_likelihood,
            max_anomaly_score, mean_anomaly_score, predicted_points, test_points
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _append_observations(conn: sqlite3.Connection, run_id: int, observations: list[dict[str, object]]) -> None:
    rows = [
        (
            run_id,
            row.get("log_entry_id"),
            row["instance"],
            row["observation_type"],
            row["severity"],
            float(row["confidence"]),
            row["summary"],
            row["evidence_json"] if isinstance(row["evidence_json"], str) else json.dumps(row["evidence_json"]),
            row["recommendation"],
        )
        for row in observations
    ]
    conn.executemany(
        """
        INSERT INTO observations(
            run_id, log_entry_id, instance, observation_type, severity, confidence,
            summary, evidence_json, recommendation
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _append_log_entries(
    conn: sqlite3.Connection,
    run_id: int,
    alert_windows: pd.DataFrame,
    host_scores: pd.DataFrame,
    observations: list[dict[str, object]],
    *,
    slm_balancer: "SlmLoadBalancer | None" = None,
) -> None:
    entries: list[tuple[object, ...]] = []
    best_model = _best_model_name(conn, run_id)
    host_score_lookup = _host_score_lookup(host_scores, best_model)
    for row in alert_windows.fillna("").to_dict("records"):
        message = f"{row.get('alertname', 'alert')} firing on {row.get('instance', '')}"
        entries.append(
            (
                run_id,
                "alert",
                row.get("start_time", ""),
                row.get("instance", ""),
                row.get("severity", ""),
                row.get("alertname", ""),
                message,
                json.dumps(row, default=str),
            )
        )

    for row in host_scores.fillna("").to_dict("records"):
        if best_model and row.get("model") != best_model:
            continue
        severity = "critical" if int(float(row.get("alert_host", 0) or 0)) else "warning"
        message = (
            f"{row.get('model', '')} scored {row.get('instance', '')} with "
            f"mean_likelihood={row.get('mean_likelihood', '')}"
        )
        entries.append(
            (
                run_id,
                "anomaly_score",
                "",
                row.get("instance", ""),
                severity,
                str(row.get("model", "")),
                message,
                json.dumps(row, default=str),
            )
        )

    for row in observations:
        entries.append(
            (
                run_id,
                "slm_observation",
                "",
                row.get("instance", ""),
                row.get("severity", ""),
                row.get("observation_type", ""),
                row.get("summary", ""),
                json.dumps(row, default=str),
            )
        )

    for entry in entries:
        cursor = conn.execute(
            """
            INSERT INTO log_entries(run_id, entry_type, timestamp, instance, severity, source, message, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            entry,
        )
        log_entry_id = int(cursor.lastrowid)
        model_output = _entry_model_output(log_entry_id, entry, host_score_lookup, best_model)
        _append_entry_model_output(conn, run_id, model_output)
        metadata_observation = _entry_metadata_observation(log_entry_id, entry)
        _append_observations(conn, run_id, [metadata_observation])
        _append_entry_slm_extraction(conn, run_id, _entry_slm_extraction(entry, metadata_observation, model_output, slm_balancer))


def _host_score_lookup(host_scores: pd.DataFrame, best_model: str) -> dict[str, dict[str, object]]:
    rows = host_scores.fillna("").to_dict("records")
    if best_model:
        rows = [row for row in rows if row.get("model") == best_model]
    return {str(row.get("instance", "")): row for row in rows}


def _entry_model_output(
    log_entry_id: int,
    entry: tuple[object, ...],
    host_score_lookup: dict[str, dict[str, object]],
    best_model: str,
) -> dict[str, object]:
    _, entry_type, _, instance, _, source, _, payload_json = entry
    payload = _loads_json(str(payload_json or "{}"))
    score = payload if entry_type == "anomaly_score" and isinstance(payload, dict) else host_score_lookup.get(str(instance or ""), {})
    output_type = "direct_anomaly_score" if entry_type == "anomaly_score" else "related_host_score"
    if not score:
        output_type = "missing_host_score"
    model = str(score.get("model", source or best_model or "")) if isinstance(score, dict) else str(best_model or "")
    return {
        "log_entry_id": log_entry_id,
        "instance": str(instance or ""),
        "model": model,
        "output_type": output_type,
        "alert_host": _int_or_none(score.get("alert_host") if isinstance(score, dict) else None),
        "max_likelihood": _float_or_none(score.get("max_likelihood") if isinstance(score, dict) else None),
        "mean_likelihood": _float_or_none(score.get("mean_likelihood") if isinstance(score, dict) else None),
        "max_anomaly_score": _float_or_none(score.get("max_anomaly_score") if isinstance(score, dict) else None),
        "mean_anomaly_score": _float_or_none(score.get("mean_anomaly_score") if isinstance(score, dict) else None),
        "predicted_points": _int_or_none(score.get("predicted_points") if isinstance(score, dict) else None),
        "test_points": _int_or_none(score.get("test_points") if isinstance(score, dict) else None),
        "evidence_json": json.dumps({"entry_type": entry_type, "best_model": best_model, "model_output": score}, default=str),
    }


def _append_entry_model_output(conn: sqlite3.Connection, run_id: int, row: dict[str, object]) -> None:
    conn.execute(
        """
        INSERT INTO entry_model_outputs(
            run_id, log_entry_id, instance, model, output_type, alert_host,
            max_likelihood, mean_likelihood, max_anomaly_score, mean_anomaly_score,
            predicted_points, test_points, evidence_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row["log_entry_id"],
            row["instance"],
            row["model"],
            row["output_type"],
            row["alert_host"],
            row["max_likelihood"],
            row["mean_likelihood"],
            row["max_anomaly_score"],
            row["mean_anomaly_score"],
            row["predicted_points"],
            row["test_points"],
            row["evidence_json"],
        ),
    )


def _entry_slm_extraction(
    entry: tuple[object, ...],
    metadata_observation: dict[str, object],
    model_output: dict[str, object],
    slm_balancer: "SlmLoadBalancer | None",
) -> dict[str, object]:
    _, entry_type, timestamp, instance, severity, source, message, payload_json = entry
    evidence = {
        "log_entry": {
            "entry_type": entry_type,
            "timestamp": timestamp,
            "instance": instance,
            "severity": severity,
            "source": source,
            "message": message,
            "payload": _loads_json(str(payload_json or "{}")),
        },
        "metadata": metadata_observation,
        "model_output": model_output,
    }
    model = "deterministic_extractor"
    raw_response = ""
    parsed = _fallback_slm_extraction(evidence)
    if slm_balancer is not None:
        try:
            response = slm_balancer.extract_log_entry(evidence)
            raw_response = response.content
            parsed = {**parsed, **parse_slm_json(response.content)}
            model = response.model
        except Exception as exc:
            raw_response = f"SLM fallback used: {exc}"
    entities = parsed.get("entities", {})
    if not isinstance(entities, dict):
        entities = {}
    return {
        "log_entry_id": int(metadata_observation["log_entry_id"]),
        "instance": str(instance or ""),
        "model": model,
        "summary": str(parsed.get("summary", metadata_observation["summary"])),
        "signal_type": str(parsed.get("signal_type", entry_type or "other")),
        "risk": str(parsed.get("risk", _risk_from_severity(str(severity or "")))),
        "recommendation": str(parsed.get("recommendation", metadata_observation["recommendation"])),
        "confidence_reason": str(parsed.get("confidence_reason", "Extracted from structured log-entry metadata and ML context.")),
        "entities_json": json.dumps(entities, default=str),
        "extraction_json": json.dumps(parsed, default=str),
        "raw_response": raw_response,
    }


def _append_entry_slm_extraction(conn: sqlite3.Connection, run_id: int, row: dict[str, object]) -> None:
    conn.execute(
        """
        INSERT INTO entry_slm_extractions(
            run_id, log_entry_id, instance, model, summary, signal_type, risk,
            recommendation, confidence_reason, entities_json, extraction_json, raw_response
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row["log_entry_id"],
            row["instance"],
            row["model"],
            row["summary"],
            row["signal_type"],
            row["risk"],
            row["recommendation"],
            row["confidence_reason"],
            row["entities_json"],
            row["extraction_json"],
            row["raw_response"],
        ),
    )


def _fallback_slm_extraction(evidence: dict[str, object]) -> dict[str, object]:
    log_entry = evidence["log_entry"]
    model_output = evidence["model_output"]
    entry_type = str(log_entry.get("entry_type", "other"))
    severity = str(log_entry.get("severity", "info"))
    payload = log_entry.get("payload", {})
    payload = payload if isinstance(payload, dict) else {}
    return {
        "summary": str(evidence["metadata"]["summary"]),
        "entities": {
            "instance": str(log_entry.get("instance", "")),
            "alertname": str(payload.get("alertname", "")),
            "model": str(model_output.get("model", "")),
            "metric": _metric_hint(payload, entry_type),
            "mountpoint": str(payload.get("mountpoint", "")),
            "device": str(payload.get("device", "")),
        },
        "signal_type": entry_type if entry_type in {"alert", "anomaly_score", "slm_observation"} else "other",
        "recommendation": str(evidence["metadata"]["recommendation"]),
        "confidence_reason": "Extracted from structured payload fields and linked ML model output.",
        "risk": _risk_from_severity(severity),
    }


def _metric_hint(payload: dict[str, object], entry_type: str) -> str:
    if entry_type == "anomaly_score":
        return "anomaly_likelihood"
    alertname = str(payload.get("alertname", ""))
    if "Disk" in alertname or "Filesystem" in alertname:
        return "disk"
    return ""


def _entry_metadata_observation(log_entry_id: int, entry: tuple[object, ...]) -> dict[str, object]:
    _, entry_type, timestamp, instance, severity, source, message, payload_json = entry
    payload = _loads_json(str(payload_json or "{}"))
    evidence = {
        "entry_type": entry_type,
        "timestamp": timestamp,
        "source": source,
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "payload": payload,
    }
    confidence = _metadata_confidence(str(entry_type), str(severity or ""))
    summary = _metadata_summary(str(entry_type), str(instance or ""), str(source or ""), str(message or ""))
    return {
        "log_entry_id": log_entry_id,
        "instance": str(instance or ""),
        "observation_type": "entry_metadata",
        "severity": str(severity or "info"),
        "confidence": confidence,
        "summary": summary,
        "evidence_json": json.dumps(evidence, default=str),
        "recommendation": _metadata_recommendation(str(entry_type), str(source or ""), str(severity or "")),
    }


def _metadata_summary(entry_type: str, instance: str, source: str, message: str) -> str:
    if entry_type == "alert":
        return f"Alert log entry from {source or 'Prometheus'} for {instance}: {message}"
    if entry_type == "anomaly_score":
        return f"ML anomaly-score log entry for {instance} from model {source}: {message}"
    if entry_type == "slm_observation":
        return f"SLM observation log entry for {instance}: {message}"
    return f"{entry_type or 'Log'} entry for {instance}: {message}"


def _metadata_recommendation(entry_type: str, source: str, severity: str) -> str:
    if entry_type == "alert":
        return "Use this alert entry as a concrete incident candidate and compare it with nearby telemetry points."
    if entry_type == "anomaly_score":
        return "Use this anomaly score as model context and verify against Prometheus points before paging."
    if entry_type == "slm_observation":
        return "Use this extracted summary as analyst context, then confirm with raw telemetry and alerts."
    if severity in {"critical", "high"}:
        return "Review this high-priority entry with adjacent telemetry and recent feedback."
    return "Keep as supporting context unless it correlates with higher-severity entries."


def _metadata_confidence(entry_type: str, severity: str) -> float:
    if entry_type == "alert":
        return 0.9
    if entry_type == "slm_observation":
        return 0.8
    if entry_type == "anomaly_score":
        return 0.7 if severity in {"critical", "high"} else 0.55
    return 0.5


def _loads_json(value: str) -> object:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _float_or_none(value: object) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        if value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _risk_from_severity(severity: str) -> str:
    if severity in {"critical", "high"}:
        return "high"
    if severity == "warning":
        return "medium"
    return "low"


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def _best_model_name(conn: sqlite3.Connection, run_id: int) -> str:
    row = conn.execute(
        "SELECT model FROM model_scores WHERE run_id = ? ORDER BY host_mean_score_auc DESC, f1 DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    return str(row[0]) if row else ""
