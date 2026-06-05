from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from src.anomaly_models import (
    build_autoencoder,
    isolation_forest_scores,
    reconstruction_errors,
    save_model,
    train_autoencoder,
)
from src.anomaly_preprocessing import (
    ANOMALY_FEATURES,
    HostDataset,
    load_prometheus_timeseries,
    make_sliding_windows,
    select_pilot_hosts,
)
from src.anomaly_scoring import calibrate_likelihood, evaluate_events, likelihood_scores


AUTOENCODER_MODELS = ["gru", "tcn", "transformer", "tsmixer"]
ALL_MODELS = AUTOENCODER_MODELS + ["isolation_forest"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Prometheus anomaly detection paper replication pilot.")
    parser.add_argument("--timeseries", default="data/server_metrics_timeseries.csv")
    parser.add_argument("--results-dir", default="results/anomaly_pilot")
    parser.add_argument("--models", default=",".join(ALL_MODELS))
    parser.add_argument("--max-hosts", type=int, default=16)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
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

    df = load_prometheus_timeseries(args.timeseries)
    host_datasets = select_pilot_hosts(df, max_hosts=args.max_hosts)
    if not host_datasets:
        raise RuntimeError("No hosts with sustained metric-threshold anomaly windows were found.")

    score_rows: list[dict[str, object]] = []
    detection_rows: list[dict[str, object]] = []
    anomaly_window_rows = _anomaly_window_rows(host_datasets)

    for host_dataset in host_datasets:
        for model_name in requested_models:
            score_row, model_detection_rows = _run_host_model(
                host_dataset,
                model_name,
                window_size=args.window_size,
                stride=args.stride,
                epochs=args.epochs,
                hidden_size=args.hidden_size,
                seed=args.seed,
                device=device,
                models_dir=models_dir,
            )
            score_rows.append(score_row)
            detection_rows.extend(model_detection_rows)

    _write_csv(results_dir / "model_scores.csv", score_rows)
    _write_csv(results_dir / "detections.csv", detection_rows)
    _write_csv(results_dir / "anomaly_windows.csv", anomaly_window_rows)
    summary = _summary(score_rows, host_datasets, requested_models, args, device)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "report.md").write_text(_report(summary, score_rows), encoding="utf-8")
    print(f"Wrote anomaly pilot outputs to {results_dir.resolve()}.")


def _run_host_model(
    host_dataset: HostDataset,
    model_name: str,
    *,
    window_size: int,
    stride: int,
    epochs: int,
    hidden_size: int,
    seed: int,
    device: str,
    models_dir: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    windows, end_indices = make_sliding_windows(host_dataset.values, window_size=window_size, stride=stride)
    window_labels = host_dataset.labels[end_indices]
    train_mask = end_indices < host_dataset.validation_start
    validation_mask = (end_indices >= host_dataset.validation_start) & (end_indices < host_dataset.test_start)
    test_mask = end_indices >= host_dataset.test_start
    train_windows = windows[train_mask]
    validation_windows = windows[validation_mask]

    if model_name == "isolation_forest":
        window_errors = isolation_forest_scores(train_windows, windows, seed=seed)
        train_loss = 0.0
        validation_loss = float(np.mean(window_errors[validation_mask])) if np.any(validation_mask) else 0.0
    else:
        model = build_autoencoder(
            model_name,
            n_features=host_dataset.values.shape[1],
            window_size=window_size,
            hidden_size=hidden_size,
        )
        losses = train_autoencoder(
            model,
            train_windows,
            validation_windows,
            epochs=epochs,
            seed=seed,
            device=device,
        )
        window_errors = reconstruction_errors(model, windows, device=device)
        save_model(model, models_dir / f"{_safe_name(host_dataset.instance)}_{model_name}.pt")
        train_loss = losses["train_loss"]
        validation_loss = losses["validation_reconstruction_error"]

    calibration = calibrate_likelihood(
        window_errors[validation_mask],
        window_labels[validation_mask],
        candidate_long_windows=[64, 96, 128, 192, 256],
        candidate_short_windows=[3, 6, 12, 24],
        candidate_thresholds=[0.95, 0.975, 0.99, 0.995, 0.999],
    )
    likelihood = likelihood_scores(
        window_errors,
        long_window=calibration.long_window,
        short_window=calibration.short_window,
    )
    window_predictions = likelihood >= calibration.threshold
    point_predictions = _window_predictions_to_points(window_predictions, end_indices, len(host_dataset.values))
    metrics = evaluate_events(point_predictions, host_dataset.anomaly_windows, test_start=host_dataset.test_start)

    score_row = {
        "instance": host_dataset.instance,
        "model": model_name,
        "precision": round(metrics.precision, 6),
        "recall": round(metrics.recall, 6),
        "f1": round(metrics.f1, 6),
        "nab_score": round(metrics.nab_score, 6),
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "mean_detection_delay": "" if metrics.mean_detection_delay is None else round(metrics.mean_detection_delay, 6),
        "long_window": calibration.long_window,
        "short_window": calibration.short_window,
        "threshold": calibration.threshold,
        "validation_f1": round(calibration.validation_f1, 6),
        "train_loss": round(train_loss, 6),
        "validation_reconstruction_error": round(validation_loss, 6),
        "test_windows": sum(1 for start, end in host_dataset.anomaly_windows if end >= host_dataset.test_start),
    }
    detection_rows = _detection_rows(host_dataset, model_name, end_indices[test_mask], likelihood[test_mask], window_errors[test_mask], window_predictions[test_mask])
    return score_row, detection_rows


def _window_predictions_to_points(window_predictions: np.ndarray, end_indices: np.ndarray, n_points: int) -> np.ndarray:
    points = np.zeros(n_points, dtype=bool)
    points[end_indices[window_predictions]] = True
    return points


def _detection_rows(
    host_dataset: HostDataset,
    model_name: str,
    end_indices: np.ndarray,
    likelihood: np.ndarray,
    errors: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, score, error, prediction in zip(end_indices, likelihood, errors, predictions):
        rows.append(
            {
                "instance": host_dataset.instance,
                "model": model_name,
                "timestamp": host_dataset.timestamps[int(idx)],
                "point_index": int(idx),
                "anomaly_label": int(host_dataset.labels[int(idx)]),
                "reconstruction_error": round(float(error), 8),
                "likelihood": round(float(score), 8),
                "prediction": int(bool(prediction)),
            }
        )
    return rows


def _anomaly_window_rows(host_datasets: list[HostDataset]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for host_dataset in host_datasets:
        for window_id, (start, end) in enumerate(host_dataset.anomaly_windows, start=1):
            rows.append(
                {
                    "instance": host_dataset.instance,
                    "window_id": window_id,
                    "start_index": start,
                    "end_index": end,
                    "start_timestamp": host_dataset.timestamps[start],
                    "end_timestamp": host_dataset.timestamps[end],
                    "split": "test" if end >= host_dataset.test_start else "train_validation",
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    score_rows: list[dict[str, object]],
    host_datasets: list[HostDataset],
    requested_models: list[str],
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    best_by_model: dict[str, float] = {}
    for model_name in requested_models:
        model_scores = [float(row["nab_score"]) for row in score_rows if row["model"] == model_name]
        best_by_model[model_name] = round(float(np.mean(model_scores)), 6) if model_scores else 0.0
    ranking = sorted(best_by_model.items(), key=lambda item: item[1], reverse=True)
    return {
        "timeseries": args.timeseries,
        "features": ANOMALY_FEATURES,
        "host_count": len(host_datasets),
        "models": requested_models,
        "window_size": args.window_size,
        "stride": args.stride,
        "epochs": args.epochs,
        "device": device,
        "mean_nab_score_by_model": best_by_model,
        "ranking": [{"rank": idx + 1, "model": model, "mean_nab_score": score} for idx, (model, score) in enumerate(ranking)],
        "note": "Metric-threshold labels are weak labels derived from training-only robust thresholds and sustained windows.",
    }


def _report(summary: dict[str, object], score_rows: list[dict[str, object]]) -> str:
    ranking = summary["ranking"]
    best = ranking[0] if ranking else {"model": "none", "mean_nab_score": 0}
    lines = [
        "# Prometheus Anomaly Pilot Report",
        "",
        f"Best mean NAB-style score: `{best['model']}` ({best['mean_nab_score']}).",
        "",
        "## Model Ranking",
        "",
        "| Rank | Model | Mean NAB-style score |",
        "| --- | --- | ---: |",
    ]
    for row in ranking:
        lines.append(f"| {row['rank']} | `{row['model']}` | {row['mean_nab_score']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Labels are weak metric-threshold windows, not confirmed incidents.",
            "- Calibration uses validation data only; test statistics are not used for thresholds.",
            f"- Evaluated {summary['host_count']} hosts across {len(summary['models'])} models.",
            f"- Produced {len(score_rows)} host/model score rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def _safe_name(value: str) -> str:
    return value.replace(":", "_").replace("/", "_").replace("\\", "_")


def _resolve_device(requested_device: str) -> str:
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false in this environment.")
        return "cuda"
    if requested_device == "auto" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


if __name__ == "__main__":
    main()
