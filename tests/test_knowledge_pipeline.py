from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.knowledge_db import ingest_alert_anomaly_results
from src.telemetry_summaries import generate_observations


class KnowledgePipelineTests(unittest.TestCase):
    def test_observations_and_database_ingest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = root / "metrics.csv"
            alerts = root / "alerts.csv"
            results = root / "results"
            results.mkdir()
            db_path = root / "knowledge.db"

            _write_fixture_metrics(metrics)
            _write_fixture_alerts(alerts)
            _write_fixture_results(results)

            observations = generate_observations(metrics, alerts, results / "host_scores.csv", top_n=2)
            self.assertEqual(len(observations), 2)
            self.assertIn("summary", observations[0])
            json.loads(observations[0]["evidence_json"])

            run_id = ingest_alert_anomaly_results(
                db_path,
                results,
                metrics_path=metrics,
                alert_windows_path=alerts,
                observations=observations,
            )
            self.assertEqual(run_id, 1)
            conn = sqlite3.connect(db_path)
            try:
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM hosts").fetchone()[0], 2)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM telemetry_points").fetchone()[0], 8)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM log_entries").fetchone()[0], 5)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 7)
                self.assertEqual(
                    conn.execute("SELECT COUNT(*) FROM observations WHERE log_entry_id IS NOT NULL").fetchone()[0],
                    5,
                )
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM entry_model_outputs").fetchone()[0], 5)
                self.assertEqual(conn.execute("SELECT COUNT(*) FROM entry_slm_extractions").fetchone()[0], 5)
            finally:
                conn.close()


def _write_fixture_metrics(path: Path) -> None:
    rows = []
    for instance, disk in [("host-alert", 98.0), ("host-ok", 20.0)]:
        for i in range(4):
            rows.append(
                {
                    "timestamp": f"2026-06-01 00:0{i}:00",
                    "instance": instance,
                    "ip": instance,
                    "rack": "rack-a",
                    "unit": "1",
                    "note": "",
                    "assigned_user": "tester",
                    "up": 1,
                    "cpu_usage_percent": 1.0,
                    "mem_used_percent": 10.0,
                    "disk_used_percent": disk,
                    "load1": 0.1,
                    "load5": 0.1,
                    "load15": 0.1,
                    "network_receive_mbps": 0.0,
                    "network_transmit_mbps": 0.0,
                    "disk_read_mb_s": 0.0,
                    "disk_write_mb_s": 0.0,
                    "uptime_days": 10.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_fixture_alerts(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "start_time": "2026-06-01 00:00:00",
                "end_time": "2026-06-01 00:05:00",
                "sample_count": 4,
                "alertname": "HostOutOfDiskSpace",
                "alertstate": "firing",
                "severity": "critical",
                "instance": "host-alert",
                "ip": "host-alert",
                "rack": "rack-a",
                "unit": "1",
                "note": "",
                "assigned_user": "tester",
                "device": "/dev/sda2",
                "mountpoint": "/",
            }
        ]
    ).to_csv(path, index=False)


def _write_fixture_results(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "model": "tsmixer",
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "balanced_accuracy": 0.5,
                "host_mean_score_auc": 0.7,
                "host_average_precision": 0.6,
                "alert_host_recall": 1.0,
                "healthy_host_false_positive_rate": 0.0,
            }
        ]
    ).to_csv(path / "model_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "tsmixer",
                "instance": "host-alert",
                "alert_host": 1,
                "max_likelihood": 0.99,
                "mean_likelihood": 0.9,
                "max_anomaly_score": 5.0,
                "mean_anomaly_score": 3.0,
                "predicted_points": 3,
                "test_points": 4,
            },
            {
                "model": "tsmixer",
                "instance": "host-ok",
                "alert_host": 0,
                "max_likelihood": 0.2,
                "mean_likelihood": 0.1,
                "max_anomaly_score": 1.0,
                "mean_anomaly_score": 0.5,
                "predicted_points": 0,
                "test_points": 4,
            },
        ]
    ).to_csv(path / "host_scores.csv", index=False)
    pd.DataFrame(
        [
            {
                "start_time": "2026-06-01 00:00:00",
                "end_time": "2026-06-01 00:05:00",
                "alertname": "HostOutOfDiskSpace",
                "severity": "critical",
                "instance": "host-alert",
                "device": "/dev/sda2",
                "mountpoint": "/",
                "sample_count": 4,
            }
        ]
    ).to_csv(path / "alert_windows.csv", index=False)
    (path / "summary.json").write_text("{}", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
