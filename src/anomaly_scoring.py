from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from scipy.special import ndtr as _normal_cdf_array
except ImportError:  # pragma: no cover - scipy is installed with scikit-learn in the project venv.
    _normal_cdf_array = None


@dataclass(frozen=True)
class Calibration:
    long_window: int
    short_window: int
    threshold: float
    validation_f1: float


@dataclass(frozen=True)
class EventMetrics:
    precision: float
    recall: float
    f1: float
    nab_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    mean_detection_delay: float | None


def likelihood_scores(errors: np.ndarray, *, long_window: int, short_window: int) -> np.ndarray:
    errors = np.asarray(errors, dtype=np.float64)
    scores = np.zeros_like(errors, dtype=np.float32)
    if len(errors) == 0:
        return scores
    long_mean, long_std = _rolling_mean_std(errors, long_window)
    short_mean, _ = _rolling_mean_std(errors, short_window)
    valid = long_std >= 1e-9
    z_scores = np.zeros_like(errors)
    z_scores[valid] = (short_mean[valid] - long_mean[valid]) / long_std[valid]
    if _normal_cdf_array is not None:
        scores[valid] = _normal_cdf_array(z_scores[valid]).astype(np.float32)
    else:
        scores[valid] = np.array([_normal_cdf(float(value)) for value in z_scores[valid]], dtype=np.float32)
    return scores


def calibrate_likelihood(
    errors: np.ndarray,
    labels: np.ndarray,
    *,
    candidate_long_windows: Iterable[int],
    candidate_short_windows: Iterable[int],
    candidate_thresholds: Iterable[float],
) -> Calibration:
    best = Calibration(long_window=64, short_window=3, threshold=0.99, validation_f1=-1.0)
    for long_window in candidate_long_windows:
        for short_window in candidate_short_windows:
            if short_window >= long_window:
                continue
            scores = likelihood_scores(errors, long_window=long_window, short_window=short_window)
            for threshold in candidate_thresholds:
                predictions = scores >= threshold
                f1 = point_f1(predictions, labels)
                if f1 > best.validation_f1:
                    best = Calibration(long_window, short_window, threshold, f1)
    return best


def point_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    tp = int(np.sum(predictions & labels))
    fp = int(np.sum(predictions & ~labels))
    fn = int(np.sum(~predictions & labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_events(
    predictions: np.ndarray,
    anomaly_windows: list[tuple[int, int]],
    *,
    test_start: int,
) -> EventMetrics:
    test_predictions = np.flatnonzero(predictions)
    test_windows = [(max(start, test_start), end) for start, end in anomaly_windows if end >= test_start]
    matched_predictions: set[int] = set()
    delays: list[int] = []
    matched_window_scores: list[float] = []
    true_positives = 0

    for start, end in test_windows:
        hits = [idx for idx in test_predictions if start <= idx <= end and idx not in matched_predictions]
        if hits:
            first_hit = min(hits)
            matched_predictions.add(first_hit)
            true_positives += 1
            delay = first_hit - start
            delays.append(delay)
            width = max(1, end - start + 1)
            matched_window_scores.append(max(0.0, 1.0 - delay / width))

    false_negatives = len(test_windows) - true_positives
    false_positives = int(sum(1 for idx in test_predictions if not any(start <= idx <= end for start, end in test_windows)))
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    nab_score = _approximate_normalized_nab(false_positives, false_negatives, matched_window_scores, test_windows)
    return EventMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        nab_score=nab_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        mean_detection_delay=float(np.mean(delays)) if delays else None,
    )


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))
    indices = np.arange(len(values))
    starts = np.maximum(0, indices - window + 1)
    counts = indices - starts + 1
    sums = prefix[indices + 1] - prefix[starts]
    sq_sums = prefix_sq[indices + 1] - prefix_sq[starts]
    means = sums / counts
    variance = np.zeros_like(values, dtype=np.float64)
    valid = counts > 1
    variance[valid] = (sq_sums[valid] - (sums[valid] * sums[valid] / counts[valid])) / (counts[valid] - 1)
    variance = np.maximum(variance, 0.0)
    return means, np.sqrt(variance)


def _approximate_normalized_nab(
    false_positives: int,
    false_negatives: int,
    matched_window_scores: list[float],
    windows: list[tuple[int, int]],
) -> float:
    if not windows:
        return 0.0 if false_positives == 0 else -100.0
    score = sum(matched_window_scores)
    score -= 0.11 * false_positives
    score -= 1.0 * false_negatives
    ideal = max(1, len(windows))
    return 100.0 * score / ideal
