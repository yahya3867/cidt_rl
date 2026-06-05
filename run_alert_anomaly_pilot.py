from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_anomaly_pilot import ALL_MODELS, AUTOENCODER_MODELS
from src.anomaly_models import (
    build_autoencoder,
    isolation_forest_scores,
    reconstruction_errors,
    save_model,
    train_autoencoder,
)
from src.anomaly_preprocessing import make_sliding_windows
from src.anomaly_scoring import calibrate_likelihood, likelihood_scores, point_f1


HISTORY_FEATURES = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run alert-window anomaly detection on CIDT Prometheus history.")
    parser.add_argument("--metrics", default=r"C:\Users\ymasr\Downloads\cidt_metrics_history.csv")
    parser.add_argument("--alert-windows", default=r"C:\Users\ymasr\Downloads\cidt_alert_windows.csv")
    parser.add_argument("--alert-name", default="", help="Optional alertname filter, e.g. HostOutOfDiskSpace.")
    parser.add_argument("--results-dir", default="results/alert_anomaly_pilot")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-train-windows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = _resolve_device(args.device)
    results_dir = Path(args.results_dir)
    models_dir = results_dir / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    unknown = sorted(set(requested_models) - set(ALL_MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    metrics = _load_metrics(args.metrics)
    alerts = _load_alerts(args.alert_windows, alert_name=args.alert_name)
    prepared = _prepare_windows(metrics, alerts, args.window_size, args.stride)

    score_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    host_rows: list[dict[str, object]] = []

    for model_name in requested_models:
        errors, train_loss, validation_loss = _score_model(
            model_name,
            prepared,
            epochs=args.epochs,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            max_train_windows=args.max_train_windows,
            seed=args.seed,
            device=device,
            models_dir=models_dir,
        )
        likelihood = _calibrated_likelihood(errors, prepared, split="validation")
        calibration = likelihood["calibration"]
        scores = likelihood["scores"]
        predictions = scores >= calibration.threshold

        score_rows.append(
            _model_score_row(
                model_name,
                prepared,
                predictions,
                scores,
                errors,
                calibration,
                train_loss,
                validation_loss,
            )
        )
        detection_rows.extend(_detection_rows(model_name, prepared, predictions, scores, errors))
        host_rows.extend(_host_rows(model_name, prepared, predictions, scores, errors))

    _write_csv(results_dir / "model_scores.csv", score_rows)
    _write_csv(results_dir / "detections.csv", detection_rows)
    _write_csv(results_dir / "host_scores.csv", host_rows)
    _write_csv(results_dir / "alert_windows.csv", _alert_window_rows(alerts))
    summary = _summary(args, prepared, score_rows, device)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    print(f"Wrote alert anomaly pilot outputs to {results_dir.resolve()}.")


def _load_metrics(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df["up"] == 1].copy()
    df = df.dropna(subset=HISTORY_FEATURES)
    df = df.sort_values(["instance", "timestamp"]).reset_index(drop=True)
    return df


def _load_alerts(path: str, *, alert_name: str = "") -> pd.DataFrame:
    alerts = pd.read_csv(path, parse_dates=["start_time", "end_time"])
    alerts = alerts[alerts["alertstate"].str.lower() == "firing"].copy()
    if alert_name:
        alerts = alerts[alerts["alertname"] == alert_name].copy()
    if alerts.empty:
        raise RuntimeError(f"No firing alert windows found for alert filter: {alert_name or 'all alerts'}")
    return alerts


def _prepare_windows(metrics: pd.DataFrame, alerts: pd.DataFrame, window_size: int, stride: int) -> dict[str, object]:
    alert_hosts = set(alerts["instance"])
    unique_timestamps = np.array(sorted(metrics["timestamp"].unique()))
    validation_start_time = pd.Timestamp(unique_timestamps[int(len(unique_timestamps) * 0.63)])
    test_start_time = pd.Timestamp(unique_timestamps[int(len(unique_timestamps) * 0.70)])

    normal_train = metrics[(~metrics["instance"].isin(alert_hosts)) & (metrics["timestamp"] < validation_start_time)]
    feature_mean = normal_train[HISTORY_FEATURES].mean()
    feature_std = normal_train[HISTORY_FEATURES].std().replace(0, 1.0).fillna(1.0)

    all_windows: list[np.ndarray] = []
    all_end_times: list[pd.Timestamp] = []
    all_instances: list[str] = []
    all_labels: list[bool] = []
    all_alertnames: list[str] = []

    alert_lookup = _alert_lookup(alerts)
    for instance, host_frame in metrics.groupby("instance", sort=True):
        values = ((host_frame[HISTORY_FEATURES] - feature_mean) / feature_std).to_numpy(dtype=np.float32)
        windows, end_indices = make_sliding_windows(values, window_size=window_size, stride=stride)
        if len(windows) == 0:
            continue
        timestamps = host_frame["timestamp"].iloc[end_indices].to_list()
        labels, alertnames = _labels_for_instance(instance, timestamps, alert_lookup)
        all_windows.append(windows)
        all_end_times.extend(timestamps)
        all_instances.extend([instance] * len(windows))
        all_labels.extend(labels)
        all_alertnames.extend(alertnames)

    windows = np.concatenate(all_windows).astype(np.float32)
    end_times = np.array(all_end_times, dtype=object)
    labels = np.array(all_labels, dtype=bool)
    instances = np.array(all_instances, dtype=object)
    alertnames = np.array(all_alertnames, dtype=object)
    train_mask = (end_times < validation_start_time) & (~labels)
    validation_mask = (end_times >= validation_start_time) & (end_times < test_start_time)
    test_mask = end_times >= test_start_time

    return {
        "windows": windows,
        "end_times": end_times,
        "labels": labels,
        "instances": instances,
        "alertnames": alertnames,
        "train_mask": train_mask,
        "validation_mask": validation_mask,
        "test_mask": test_mask,
        "alert_hosts": alert_hosts,
        "validation_start_time": validation_start_time,
        "test_start_time": test_start_time,
        "feature_mean": feature_mean.to_dict(),
        "feature_std": feature_std.to_dict(),
        "metrics_host_count": metrics["instance"].nunique(),
        "alert_host_count": len(alert_hosts),
    }


def _score_model(
    model_name: str,
    prepared: dict[str, object],
    *,
    epochs: int,
    hidden_size: int,
    batch_size: int,
    max_train_windows: int,
    seed: int,
    device: str,
    models_dir: Path,
) -> tuple[np.ndarray, float, float]:
    windows = prepared["windows"]
    train_windows = windows[prepared["train_mask"]]
    train_windows = _sample_windows(train_windows, max_train_windows=max_train_windows, seed=seed)

    validation_windows = windows[prepared["validation_mask"]]
    if model_name == "isolation_forest":
        errors = isolation_forest_scores(train_windows, windows, seed=seed)
        validation_loss = float(np.mean(errors[prepared["validation_mask"]]))
        return errors, 0.0, validation_loss

    model = build_autoencoder(
        model_name,
        n_features=windows.shape[2],
        window_size=windows.shape[1],
        hidden_size=hidden_size,
    )
    losses = train_autoencoder(
        model,
        train_windows,
        validation_windows,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    errors = reconstruction_errors(model, windows, batch_size=batch_size, device=device)
    save_model(model, models_dir / f"{model_name}.pt")
    return errors, losses["train_loss"], losses["validation_reconstruction_error"]


def _calibrated_likelihood(errors: np.ndarray, prepared: dict[str, object], *, split: str) -> dict[str, object]:
    scores_by_params: dict[tuple[int, int], np.ndarray] = {}
    labels = prepared["labels"]
    split_mask = prepared[f"{split}_mask"]
    instances = prepared["instances"]
    best = None
    for long_window in [64, 96, 128, 192, 256]:
        for short_window in [3, 6, 12, 24]:
            if short_window >= long_window:
                continue
            scores = _grouped_likelihood(errors, instances, long_window=long_window, short_window=short_window)
            scores_by_params[(long_window, short_window)] = scores
            for threshold in [0.95, 0.975, 0.99, 0.995, 0.999]:
                f1 = point_f1(scores[split_mask] >= threshold, labels[split_mask])
                if best is None or f1 > best.validation_f1:
                    best = calibrate_likelihood(
                        errors[:1],
                        labels[:1],
                        candidate_long_windows=[long_window],
                        candidate_short_windows=[short_window],
                        candidate_thresholds=[threshold],
                    )
                    best = type(best)(long_window, short_window, threshold, f1)
    assert best is not None
    return {"calibration": best, "scores": scores_by_params[(best.long_window, best.short_window)]}


def _grouped_likelihood(errors: np.ndarray, instances: np.ndarray, *, long_window: int, short_window: int) -> np.ndarray:
    scores = np.zeros_like(errors, dtype=np.float32)
    for instance in np.unique(instances):
        mask = instances == instance
        scores[mask] = likelihood_scores(errors[mask], long_window=long_window, short_window=short_window)
    return scores


def _model_score_row(
    model_name: str,
    prepared: dict[str, object],
    predictions: np.ndarray,
    scores: np.ndarray,
    errors: np.ndarray,
    calibration,
    train_loss: float,
    validation_loss: float,
) -> dict[str, object]:
    test_mask = prepared["test_mask"]
    labels = prepared["labels"][test_mask]
    preds = predictions[test_mask]
    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    fn = int(np.sum(~preds & labels))
    tn = int(np.sum(~preds & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = (recall + specificity) / 2
    host_metrics = _host_metric_summary(prepared, predictions, scores)
    return {
        "model": model_name,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "alert_host_recall": round(host_metrics["alert_host_recall"], 6),
        "healthy_host_false_positive_rate": round(host_metrics["healthy_host_false_positive_rate"], 6),
        "host_mean_score_auc": round(host_metrics["host_mean_score_auc"], 6),
        "host_max_score_auc": round(host_metrics["host_max_score_auc"], 6),
        "host_average_precision": round(host_metrics["host_average_precision"], 6),
        "long_window": calibration.long_window,
        "short_window": calibration.short_window,
        "threshold": calibration.threshold,
        "validation_f1": round(calibration.validation_f1, 6),
        "train_loss": round(train_loss, 6),
        "validation_score_mean": round(validation_loss, 6),
        "test_error_mean": round(float(np.mean(errors[test_mask])), 6),
    }


def _host_metric_summary(prepared: dict[str, object], predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    test_mask = prepared["test_mask"]
    alert_hosts = prepared["alert_hosts"]
    host_labels: list[int] = []
    host_mean_scores: list[float] = []
    host_max_scores: list[float] = []
    host_names: list[str] = []
    for instance in np.unique(prepared["instances"][test_mask]):
        mask = test_mask & (prepared["instances"] == instance)
        host_names.append(str(instance))
        host_labels.append(int(instance in alert_hosts))
        host_mean_scores.append(float(np.mean(scores[mask])))
        host_max_scores.append(float(np.max(scores[mask])))
    alert_total = sum(host_labels)
    top_k = max(1, alert_total)
    ranked = sorted(zip(host_names, host_labels, host_mean_scores), key=lambda row: row[2], reverse=True)
    predicted_alert_hosts = {name for name, _, _ in ranked[:top_k]}
    alert_hit = sum(1 for name, label, _ in ranked if label and name in predicted_alert_hosts)
    healthy_hit = sum(1 for name, label, _ in ranked if not label and name in predicted_alert_hosts)
    healthy_total = len(host_labels) - alert_total
    return {
        "alert_host_recall": alert_hit / alert_total if alert_total else 0.0,
        "healthy_host_false_positive_rate": healthy_hit / healthy_total if healthy_total else 0.0,
        "host_mean_score_auc": _roc_auc(host_labels, host_mean_scores),
        "host_max_score_auc": _roc_auc(host_labels, host_max_scores),
        "host_average_precision": _average_precision(host_labels, host_mean_scores),
    }


def _detection_rows(
    model_name: str,
    prepared: dict[str, object],
    predictions: np.ndarray,
    scores: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    test_indices = np.flatnonzero(prepared["test_mask"])
    for idx in test_indices:
        rows.append(
            {
                "model": model_name,
                "timestamp": pd.Timestamp(prepared["end_times"][idx]).strftime("%Y-%m-%d %H:%M:%S"),
                "instance": prepared["instances"][idx],
                "alert_label": int(prepared["labels"][idx]),
                "alertname": prepared["alertnames"][idx],
                "anomaly_score": round(float(errors[idx]), 8),
                "likelihood": round(float(scores[idx]), 8),
                "prediction": int(bool(predictions[idx])),
            }
        )
    return rows


def _host_rows(
    model_name: str,
    prepared: dict[str, object],
    predictions: np.ndarray,
    scores: np.ndarray,
    errors: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    test_mask = prepared["test_mask"]
    for instance in np.unique(prepared["instances"][test_mask]):
        mask = test_mask & (prepared["instances"] == instance)
        rows.append(
            {
                "model": model_name,
                "instance": instance,
                "alert_host": int(instance in prepared["alert_hosts"]),
                "max_likelihood": round(float(np.max(scores[mask])), 8),
                "mean_likelihood": round(float(np.mean(scores[mask])), 8),
                "max_anomaly_score": round(float(np.max(errors[mask])), 8),
                "mean_anomaly_score": round(float(np.mean(errors[mask])), 8),
                "predicted_points": int(np.sum(predictions[mask])),
                "test_points": int(np.sum(mask)),
            }
        )
    return rows


def _labels_for_instance(instance: str, timestamps: list[pd.Timestamp], alert_lookup: dict[str, list[dict[str, object]]]) -> tuple[list[bool], list[str]]:
    labels: list[bool] = []
    alertnames: list[str] = []
    windows = alert_lookup.get(instance, [])
    for timestamp in timestamps:
        active = [window["alertname"] for window in windows if window["start_time"] <= timestamp <= window["end_time"]]
        labels.append(bool(active))
        alertnames.append(";".join(sorted(set(active))))
    return labels, alertnames


def _alert_lookup(alerts: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    lookup: dict[str, list[dict[str, object]]] = {}
    for row in alerts.to_dict("records"):
        lookup.setdefault(row["instance"], []).append(row)
    return lookup


def _alert_window_rows(alerts: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "start_time": row["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": row["end_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "alertname": row["alertname"],
            "severity": row["severity"],
            "instance": row["instance"],
            "device": row.get("device", ""),
            "mountpoint": row.get("mountpoint", ""),
            "sample_count": row.get("sample_count", ""),
        }
        for row in alerts.to_dict("records")
    ]


def _sample_windows(windows: np.ndarray, *, max_train_windows: int, seed: int) -> np.ndarray:
    if max_train_windows <= 0 or len(windows) <= max_train_windows:
        return windows
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(windows), size=max_train_windows, replace=False)
    return windows[np.sort(indices)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(args: argparse.Namespace, prepared: dict[str, object], score_rows: list[dict[str, object]], device: str) -> dict[str, object]:
    ranking = sorted(
        score_rows,
        key=lambda row: (float(row["host_mean_score_auc"]), float(row["f1"]), float(row["balanced_accuracy"])),
        reverse=True,
    )
    return {
        "metrics": args.metrics,
        "alert_windows": args.alert_windows,
        "alert_name_filter": args.alert_name or "all",
        "features": HISTORY_FEATURES,
        "metrics_host_count": prepared["metrics_host_count"],
        "alert_host_count": prepared["alert_host_count"],
        "window_count": int(len(prepared["windows"])),
        "train_window_count": int(np.sum(prepared["train_mask"])),
        "validation_window_count": int(np.sum(prepared["validation_mask"])),
        "test_window_count": int(np.sum(prepared["test_mask"])),
        "test_start_time": prepared["test_start_time"].strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "device": device,
        "ranking": [
            {
                "rank": index + 1,
                "model": row["model"],
                "f1": row["f1"],
                "balanced_accuracy": row["balanced_accuracy"],
                "host_mean_score_auc": row["host_mean_score_auc"],
                "host_average_precision": row["host_average_precision"],
                "alert_host_recall": row["alert_host_recall"],
                "healthy_host_false_positive_rate": row["healthy_host_false_positive_rate"],
            }
            for index, row in enumerate(ranking)
        ],
        "note": "Models train on non-alert host telemetry only; alert windows provide real labels for validation and held-out testing.",
    }


def _report(summary: dict[str, object]) -> str:
    best = summary["ranking"][0]
    lines = [
        "# Alert-Window Anomaly Pilot Report",
        "",
        f"Best model by host ranking AUC: `{best['model']}` with AUC `{best['host_mean_score_auc']}` and F1 `{best['f1']}`.",
        "",
        "| Rank | Model | Host AUC | Avg precision | F1 | Balanced accuracy | Alert-host recall | Healthy-host FP rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["ranking"]:
        lines.append(
            f"| {row['rank']} | `{row['model']}` | {row['host_mean_score_auc']} | {row['host_average_precision']} | {row['f1']} | {row['balanced_accuracy']} | {row['alert_host_recall']} | {row['healthy_host_false_positive_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Labels come from real critical Prometheus alert windows.",
            "- Training uses non-alert hosts only, so alert hosts are not learned as normal.",
            "- The alert windows are long-lived, so this evaluates alert-host separation more than incident-start early warning.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resolve_device(requested_device: str) -> str:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false in this environment.")
        return "cuda"
    if requested_device == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _roc_auc(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        return 0.0


def _average_precision(labels: list[int], scores: list[float]) -> float:
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(labels, scores))
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
