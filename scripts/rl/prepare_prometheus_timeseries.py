from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RL_COLUMNS = [
    "timestamp",
    "instance",
    "ip",
    "rack",
    "unit",
    "note",
    "assigned_user",
    "up",
    "cpu_usage_percent",
    "cpu_cores",
    "mem_available_gb",
    "mem_total_gb",
    "disk_available_gb",
    "disk_size_gb",
    "load1",
    "load5",
    "load15",
]

REQUIRED_RL_VALUES = [
    "cpu_usage_percent",
    "cpu_cores",
    "mem_available_gb",
    "mem_total_gb",
    "disk_available_gb",
    "disk_size_gb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Prometheus history for RL scheduling experiments.")
    parser.add_argument("--metrics", required=True, help="Prometheus history CSV.")
    parser.add_argument("--alert-windows", default="", help="Optional alert windows CSV.")
    parser.add_argument("--output", required=True, help="Prepared time-series CSV.")
    parser.add_argument("--bucket", default="5min", help="Pandas timestamp floor frequency.")
    parser.add_argument(
        "--alert-policy",
        choices=["mark_down", "annotate"],
        default="mark_down",
        help="How to apply alert windows to scheduling input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = pd.read_csv(args.metrics, usecols=lambda column: column in RL_COLUMNS)
    metrics["timestamp"] = pd.to_datetime(metrics["timestamp"], errors="coerce")
    metrics = metrics.dropna(subset=["timestamp", "instance", *REQUIRED_RL_VALUES]).copy()
    metrics["timestamp"] = metrics["timestamp"].dt.floor(args.bucket)
    metrics["user"] = metrics.get("assigned_user", "")

    prepared = (
        metrics.sort_values(["timestamp", "instance"])
        .drop_duplicates(subset=["timestamp", "instance"], keep="last")
        .sort_values(["timestamp", "instance"])
        .reset_index(drop=True)
    )

    alert_count = 0
    if args.alert_windows:
        alert_count = _apply_alert_windows(prepared, Path(args.alert_windows), args.alert_policy)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output, index=False)
    print(
        {
            "output": str(output),
            "rows": int(len(prepared)),
            "timestamps": int(prepared["timestamp"].nunique()),
            "instances": int(prepared["instance"].nunique()),
            "alert_windows_applied": alert_count,
            "alert_policy": args.alert_policy,
        }
    )


def _apply_alert_windows(prepared: pd.DataFrame, alert_windows_path: Path, policy: str) -> int:
    alerts = pd.read_csv(alert_windows_path, parse_dates=["start_time", "end_time"]).fillna("")
    applied = 0
    prepared["note"] = prepared["note"].fillna("")
    for row in alerts.to_dict("records"):
        instance = str(row.get("instance", ""))
        if not instance:
            continue
        mask = (
            (prepared["instance"] == instance)
            & (prepared["timestamp"] >= row["start_time"])
            & (prepared["timestamp"] <= row["end_time"])
        )
        if not bool(mask.any()):
            continue
        applied += 1
        alert_label = str(row.get("alertname", "alert"))
        if policy == "mark_down":
            prepared.loc[mask, "up"] = 0
        prepared.loc[mask, "note"] = prepared.loc[mask, "note"].astype(str).map(
            lambda note, label=alert_label: f"{note}; {label}".strip("; ")
        )
    return applied


if __name__ == "__main__":
    main()
