from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ANOMALY_FEATURES = [
    "cpu_usage_percent",
    "mem_used_ratio",
    "disk_used_ratio",
    "load1_per_core",
    "load5_per_core",
    "load15_per_core",
]


@dataclass(frozen=True)
class HostDataset:
    instance: str
    timestamps: list[str]
    values: np.ndarray
    labels: np.ndarray
    anomaly_windows: list[tuple[int, int]]
    train_end: int
    validation_start: int
    test_start: int
    feature_mean: np.ndarray
    feature_std: np.ndarray


def load_prometheus_timeseries(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["up"] == 1].copy()
    df = df.sort_values(["instance", "timestamp"]).reset_index(drop=True)

    df["mem_used_ratio"] = 1.0 - (df["mem_available_gb"] / df["mem_total_gb"])
    df["disk_used_ratio"] = 1.0 - (df["disk_available_gb"] / df["disk_size_gb"])
    for load_col in ("load1", "load5", "load15"):
        df[f"{load_col}_per_core"] = df[load_col] / df["cpu_cores"]

    df[ANOMALY_FEATURES] = df[ANOMALY_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=ANOMALY_FEATURES)
    return df


def build_host_dataset(
    host_frame: pd.DataFrame,
    *,
    min_window_len: int = 3,
    merge_gap: int = 1,
) -> HostDataset:
    host_frame = _ensure_derived_features(host_frame).sort_values("timestamp").reset_index(drop=True)
    instance = str(host_frame["instance"].iloc[0])
    raw_values = host_frame[ANOMALY_FEATURES].to_numpy(dtype=np.float32)
    n_rows = len(host_frame)
    train_end = int(n_rows * 0.63)
    validation_start = train_end
    test_start = int(n_rows * 0.70)

    train_values = raw_values[:train_end]
    labels = weak_metric_labels(raw_values, train_values)
    anomaly_windows = collapse_anomaly_windows(labels, min_window_len=min_window_len, merge_gap=merge_gap)

    feature_mean = train_values.mean(axis=0)
    feature_std = train_values.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    values = ((raw_values - feature_mean) / feature_std).astype(np.float32)

    return HostDataset(
        instance=instance,
        timestamps=[ts.strftime("%Y-%m-%d %H:%M:%S") for ts in host_frame["timestamp"]],
        values=values,
        labels=labels,
        anomaly_windows=anomaly_windows,
        train_end=train_end,
        validation_start=validation_start,
        test_start=test_start,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def weak_metric_labels(values: np.ndarray, train_values: np.ndarray) -> np.ndarray:
    labels = np.zeros(values.shape[0], dtype=bool)
    medians = np.nanmedian(train_values, axis=0)
    mad = np.nanmedian(np.abs(train_values - medians), axis=0)
    std = np.nanstd(train_values, axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, std)
    scale = np.where(scale > 1e-9, scale, 1.0)

    for feature_index, feature_name in enumerate(ANOMALY_FEATURES):
        threshold_width = 6.0 if feature_name.startswith(("cpu", "load")) else 5.0
        labels |= values[:, feature_index] > medians[feature_index] + threshold_width * scale[feature_index]

    return labels


def _ensure_derived_features(host_frame: pd.DataFrame) -> pd.DataFrame:
    host_frame = host_frame.copy()
    if "mem_used_ratio" not in host_frame:
        host_frame["mem_used_ratio"] = 1.0 - (host_frame["mem_available_gb"] / host_frame["mem_total_gb"])
    if "disk_used_ratio" not in host_frame:
        host_frame["disk_used_ratio"] = 1.0 - (host_frame["disk_available_gb"] / host_frame["disk_size_gb"])
    for load_col in ("load1", "load5", "load15"):
        derived_col = f"{load_col}_per_core"
        if derived_col not in host_frame:
            host_frame[derived_col] = host_frame[load_col] / host_frame["cpu_cores"]
    return host_frame


def collapse_anomaly_windows(
    labels: np.ndarray,
    *,
    min_window_len: int = 3,
    merge_gap: int = 1,
) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_anomaly in enumerate(labels):
        if is_anomaly and start is None:
            start = idx
        is_last = idx == len(labels) - 1
        if start is not None and (not is_anomaly or is_last):
            end = idx if is_anomaly and is_last else idx - 1
            if end - start + 1 >= min_window_len:
                windows.append((start, end))
            start = None

    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start - merged[-1][1] - 1 <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def select_pilot_hosts(
    df: pd.DataFrame,
    *,
    max_hosts: int | None = None,
    min_test_windows: int = 1,
) -> list[HostDataset]:
    datasets: list[HostDataset] = []
    for _, host_frame in df.groupby("instance"):
        dataset = build_host_dataset(host_frame)
        test_windows = [window for window in dataset.anomaly_windows if window[1] >= dataset.test_start]
        if len(test_windows) >= min_test_windows:
            datasets.append(dataset)

    datasets.sort(
        key=lambda ds: (
            sum(1 for start, end in ds.anomaly_windows if end >= ds.test_start),
            sum(end - start + 1 for start, end in ds.anomaly_windows if end >= ds.test_start),
        ),
        reverse=True,
    )
    if max_hosts is not None:
        return datasets[:max_hosts]
    return datasets


def make_sliding_windows(values: np.ndarray, *, window_size: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    if len(values) < window_size:
        return np.empty((0, window_size, values.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)
    starts = np.arange(0, len(values) - window_size + 1, stride, dtype=np.int64)
    windows = np.stack([values[start : start + window_size] for start in starts]).astype(np.float32)
    ends = starts + window_size - 1
    return windows, ends
