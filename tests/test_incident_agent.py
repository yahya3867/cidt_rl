from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.incident_agent import insert_incident_report, load_incident_evidence
from src.knowledge_db import initialize_database


class IncidentAgentTests(unittest.TestCase):
    def test_load_evidence_and_insert_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "k.db"
            initialize_database(db)
            conn = sqlite3.connect(db)
            try:
                conn.execute("INSERT INTO model_runs(id, results_dir) VALUES (1, 'r')")
                conn.execute("INSERT INTO model_scores(run_id, model, f1, host_mean_score_auc) VALUES (1, 'm', 0.2, 0.9)")
                conn.execute(
                    "INSERT INTO host_scores(run_id, model, instance, alert_host, mean_likelihood) VALUES (1, 'm', 'host-a', 1, 0.8)"
                )
                conn.execute(
                    "INSERT INTO observations(run_id, instance, observation_type, severity, confidence, summary, evidence_json, recommendation) VALUES (1, 'host-a', 'alert_context', 'critical', 0.8, 'summary', ?, 'fix')",
                    (json.dumps({"x": 1}),),
                )
                conn.execute(
                    "INSERT INTO alert_windows(instance, alertname, severity) VALUES ('host-a', 'HostOutOfDiskSpace', 'critical')"
                )
                conn.execute(
                    "INSERT INTO log_entries(id, run_id, entry_type, instance, severity, message) VALUES (10, 1, 'alert', 'host-a', 'critical', 'alert message')"
                )
                conn.commit()
            finally:
                conn.close()
            evidence = load_incident_evidence(db, "host-a")
            self.assertEqual(evidence["best_model"], "m")
            entry_evidence = load_incident_evidence(db, log_entry_id=10)
            self.assertEqual(entry_evidence["instance"], "host-a")
            report_id = insert_incident_report(
                db,
                {
                    "run_id": 1,
                    "log_entry_id": 10,
                    "instance": "host-a",
                    "title": "t",
                    "summary": "s",
                    "likely_root_cause": "r",
                    "recommendations": "[]",
                },
            )
            self.assertEqual(report_id, 1)


if __name__ == "__main__":
    unittest.main()
