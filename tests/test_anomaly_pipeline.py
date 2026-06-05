from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.anomaly_preprocessing import build_host_dataset, collapse_anomaly_windows
from src.anomaly_scoring import likelihood_scores


class AnomalyPipelineTests(unittest.TestCase):
    def test_normalization_uses_training_statistics_only(self) -> None:
        rows = []
        for idx in range(100):
            value = 1.0 if idx < 63 else 100.0
            rows.append(
                {
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=15 * idx),
                    "instance": "host-a",
                    "cpu_usage_percent": value,
                    "mem_available_gb": 90.0,
                    "mem_total_gb": 100.0,
                    "disk_available_gb": 900.0,
                    "disk_size_gb": 1000.0,
                    "load1": 1.0,
                    "load5": 1.0,
                    "load15": 1.0,
                    "cpu_cores": 10.0,
                }
            )
        dataset = build_host_dataset(pd.DataFrame(rows))
        self.assertAlmostEqual(float(dataset.feature_mean[0]), 1.0)
        self.assertGreater(float(dataset.values[-1, 0]), 50.0)

    def test_window_generation_collapses_sustained_spikes(self) -> None:
        labels = np.array([False, True, True, False, True, True, True, False, True, False])
        self.assertEqual(collapse_anomaly_windows(labels, min_window_len=2, merge_gap=1), [(1, 6)])

    def test_likelihood_scores_are_stable_shape(self) -> None:
        errors = np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0], dtype=np.float32)
        scores = likelihood_scores(errors, long_window=4, short_window=2)
        self.assertEqual(scores.shape, errors.shape)
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertGreater(float(scores[-1]), float(scores[0]))


if __name__ == "__main__":
    unittest.main()
