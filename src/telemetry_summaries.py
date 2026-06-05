from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .slm_client import SlmLoadBalancer, parse_slm_json


SUMMARY_FEATURES = [
    "cpu_usage_percent",
    "mem_used_percent",
    "disk_used_percent",
    "load1",
    "load5",
    "load15",
    "network_receive_mbps",
    "network_transmit_mbps",
    "disk_read_mb_s",
    "disk_write_mb_s",
    "uptime_days",
]


@dataclass(frozen=True)
class Observation:
    instance: str
    observation_type: str
    severity: str
    confidence: float
    summary: str
    evidence_json: str
    recommendation: str


def generate_observations(
    metrics_path: str | Path,
    alert_windows_path: str | Path,
    host_scores_path: str | Path,
    *,
    top_n: int = 25,
    slm_balancer: SlmLoadBalancer | None = None,
) -> list[dict[str, object]]:
    metrics = pd.read_csv(metrics_path, parse_dates=["timestamp"])
    alerts = pd.read_csv(alert_windows_path, parse_dates=["start_time", "end_time"])
    host_scores = pd.read_csv(host_scores_path)
    metrics = metrics[(metrics["up"] == 1) & metrics[SUMMARY_FEATURES].notna().all(axis=1)].copy()

    best_model = _best_model_from_scores(host_scores_path, host_scores)
    model_scores = host_scores[host_scores["model"] == best_model].copy()
    model_scores = model_scores.sort_values(["alert_host", "mean_likelihood"], ascending=[False, False]).head(top_n)

    baseline = _baseline_profile(metrics, set(alerts["instance"]))
    rows: list[dict[str, object]] = []
    for score_row in model_scores.to_dict("records"):
        instance = str(score_row["instance"])
        host_metrics = metrics[metrics["instance"] == instance]
        if host_metrics.empty:
            continue
        host_alerts = alerts[alerts["instance"] == instance]
        profile = _host_profile(host_metrics, baseline)
        drivers = profile["drivers"]
        alertnames = sorted(set(host_alerts["alertname"])) if not host_alerts.empty else []
        severity = "critical" if int(score_row["alert_host"]) else ("warning" if float(score_row["mean_likelihood"]) >= 0.7 else "info")
        observation_type = "alert_context" if alertnames else "model_anomaly"
        evidence = {
            "model": best_model,
            "mean_likelihood": float(score_row["mean_likelihood"]),
            "max_likelihood": float(score_row["max_likelihood"]),
            "alertnames": alertnames,
            "top_metric_drivers": drivers,
            "alert_windows": _alert_evidence(host_alerts),
        }
        summary = _summary_text(instance, best_model, score_row, drivers, alertnames)
        recommendation = _recommendation(alertnames, drivers)
        slm_model = ""
        slm_raw = ""
        if slm_balancer is not None:
            try:
                slm_response = slm_balancer.summarize_observation({"instance": instance, **evidence})
                slm_parsed = parse_slm_json(slm_response.content)
                summary = str(slm_parsed.get("summary", summary))
                recommendation = str(slm_parsed.get("recommendation", recommendation))
                slm_model = slm_response.model
                slm_raw = slm_response.content
            except Exception as exc:
                slm_raw = f"SLM fallback used: {exc}"
        rows.append(
            {
                "instance": instance,
                "observation_type": observation_type,
                "severity": severity,
                "confidence": round(float(score_row["mean_likelihood"]), 6),
                "summary": summary,
                "evidence_json": json.dumps({**evidence, "slm_model": slm_model, "slm_raw": slm_raw}, default=str),
                "recommendation": recommendation,
            }
        )
    return rows


def write_observations(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _best_model_from_scores(host_scores_path: str | Path, host_scores: pd.DataFrame) -> str:
    model_scores_path = Path(host_scores_path).parent / "model_scores.csv"
    if model_scores_path.exists():
        model_scores = pd.read_csv(model_scores_path)
        if "host_mean_score_auc" in model_scores:
            ranked = model_scores.sort_values(["host_mean_score_auc", "f1"], ascending=False)
            return str(ranked.iloc[0]["model"])
    alert_hosts = host_scores[host_scores["alert_host"] == 1]
    if alert_hosts.empty:
        return str(host_scores.groupby("model")["mean_likelihood"].mean().sort_values(ascending=False).index[0])
    ranked = alert_hosts.groupby("model")["mean_likelihood"].mean().sort_values(ascending=False)
    return str(ranked.index[0])


def _baseline_profile(metrics: pd.DataFrame, alert_hosts: set[str]) -> pd.DataFrame:
    normal = metrics[~metrics["instance"].isin(alert_hosts)]
    return normal[SUMMARY_FEATURES].agg(["mean", "std"]).replace(0, np.nan)


def _host_profile(host_metrics: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, object]:
    means = host_metrics[SUMMARY_FEATURES].mean()
    z_scores = ((means - baseline.loc["mean"]) / baseline.loc["std"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    drivers = [
        {
            "feature": feature,
            "host_mean": round(float(means[feature]), 6),
            "normal_mean": round(float(baseline.loc["mean", feature]), 6),
            "z_score": round(float(z_scores[feature]), 6),
        }
        for feature in z_scores.abs().sort_values(ascending=False).head(4).index
    ]
    return {"drivers": drivers}


def _summary_text(instance: str, model: str, score_row: dict[str, object], drivers: list[dict[str, object]], alertnames: list[str]) -> str:
    driver_text = ", ".join(f"{driver['feature']} z={driver['z_score']}" for driver in drivers[:3])
    if alertnames:
        return (
            f"{instance} has active critical alert context ({'; '.join(alertnames)}) and was scored by {model}. "
            f"Main telemetry differences: {driver_text}."
        )
    return (
        f"{instance} is not in the alert-window labels but {model} assigned elevated anomaly context. "
        f"Main telemetry differences: {driver_text}."
    )


def _recommendation(alertnames: list[str], drivers: list[dict[str, object]]) -> str:
    driver_names = {driver["feature"] for driver in drivers[:3]}
    if any("Disk" in alert for alert in alertnames) or "disk_used_percent" in driver_names:
        return "Check filesystem capacity, mount health, stale NFS mounts, and large recent writes."
    if any("Filesystem" in alert for alert in alertnames) or {"disk_read_mb_s", "disk_write_mb_s"} & driver_names:
        return "Inspect mountpoints, NFS device errors, disk I/O saturation, and storage reachability."
    if {"network_receive_mbps", "network_transmit_mbps"} & driver_names:
        return "Inspect network throughput, interface errors, and service traffic patterns."
    return "Review host telemetry, recent changes, and related alerts before paging."


def _alert_evidence(alerts: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "alertname": row["alertname"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "device": row.get("device", ""),
            "mountpoint": row.get("mountpoint", ""),
        }
        for row in alerts.to_dict("records")
    ]
