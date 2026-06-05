from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.knowledge_db import ingest_alert_anomaly_results
from src.slm_client import DEFAULT_SLM_WORKERS, SlmLoadBalancer
from src.telemetry_summaries import generate_observations, write_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the CIDT anomaly knowledge database from experiment artifacts.")
    parser.add_argument("--metrics", default=r"C:\Users\ymasr\Downloads\cidt_metrics_history.csv")
    parser.add_argument("--alert-windows", default=r"C:\Users\ymasr\Downloads\cidt_alert_windows.csv")
    parser.add_argument("--results-dir", default="results/alert_anomaly_pilot")
    parser.add_argument("--db-path", default="results/knowledge/cidt_anomaly_knowledge.db")
    parser.add_argument("--observations-csv", default="results/knowledge/observations.csv")
    parser.add_argument("--top-observations", type=int, default=25)
    parser.add_argument("--reset", action="store_true", help="Delete the existing SQLite DB before loading this run.")
    parser.add_argument("--use-slm", action="store_true", help="Use LM Studio SLM workers for summaries.")
    parser.add_argument("--slm-base-url", default="http://127.0.0.1:1234/v1")
    parser.add_argument("--slm-models", default=",".join(DEFAULT_SLM_WORKERS))
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if args.reset and db_path.exists():
        db_path.unlink()

    slm_balancer = None
    if args.use_slm:
        slm_balancer = SlmLoadBalancer(
            base_url=args.slm_base_url,
            models=[model.strip() for model in args.slm_models.split(",") if model.strip()],
        )

    observations = generate_observations(
        args.metrics,
        args.alert_windows,
        Path(args.results_dir) / "host_scores.csv",
        top_n=args.top_observations,
        slm_balancer=slm_balancer,
    )
    write_observations(args.observations_csv, observations)
    run_id = ingest_alert_anomaly_results(
        db_path,
        args.results_dir,
        metrics_path=args.metrics,
        alert_windows_path=args.alert_windows,
        observations=observations,
        slm_balancer=slm_balancer,
    )
    counts = _table_counts(str(db_path))
    print(f"Wrote knowledge DB to {db_path.resolve()} for run_id={run_id}.")
    print(f"Wrote observations to {Path(args.observations_csv).resolve()}.")
    print(counts)


def _table_counts(db_path: str) -> dict[str, int]:
    tables = [
        "hosts",
        "telemetry_points",
        "alert_windows",
        "model_runs",
        "model_scores",
        "host_scores",
        "log_entries",
        "observations",
        "entry_model_outputs",
        "entry_slm_extractions",
    ]
    with sqlite3.connect(db_path) as conn:
        return {table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) for table in tables}


if __name__ == "__main__":
    main()
