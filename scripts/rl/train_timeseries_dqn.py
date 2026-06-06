from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl_scheduling.evaluate import summarize
from src.rl_scheduling.timeseries_env import CidtTimeseriesDqnEnv, load_timeseries_snapshots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one DQN policy across Prometheus time-series snapshots.")
    parser.add_argument("--timeseries", default="results/rl_cidt_metrics_history/prepared_timeseries.csv")
    parser.add_argument("--scenario", default="normal")
    parser.add_argument("--job-count", type=int, default=48)
    parser.add_argument("--train-snapshots", type=int, default=512, help="0 means all snapshots.")
    parser.add_argument("--eval-snapshots", type=int, default=100)
    parser.add_argument("--eval-job-seeds", default="42,43,44")
    parser.add_argument("--timesteps", type=int, default=250000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--buffer-size", type=int, default=150000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--results-dir", default="results/rl_cidt_metrics_history/timeseries_dqn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_snapshots = load_timeseries_snapshots(args.timeseries, max_snapshots=args.train_snapshots)
    eval_snapshots = load_timeseries_snapshots(args.timeseries, max_snapshots=args.eval_snapshots)
    env = Monitor(
        CidtTimeseriesDqnEnv(
            train_snapshots,
            scenario=args.scenario,
            job_count=args.job_count,
            candidate_count=args.candidate_count,
            seed=args.seed,
        )
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=0.96,
        train_freq=4,
        target_update_interval=args.target_update_interval,
        exploration_fraction=0.35,
        exploration_initial_eps=0.95,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    model_path = results_dir / "timeseries_dqn_model.zip"
    model.save(model_path)

    eval_job_seeds = [int(seed.strip()) for seed in args.eval_job_seeds.split(",") if seed.strip()]
    trial_rows = _evaluate_policy(model, eval_snapshots, args.scenario, args.job_count, args.candidate_count, eval_job_seeds)
    summary_rows = _aggregate_rows(trial_rows)
    _write_csv(results_dir / "eval_trials.csv", trial_rows)
    _write_csv(results_dir / "eval_summary.csv", summary_rows)

    summary = {
        "timeseries": args.timeseries,
        "scenario": args.scenario,
        "job_count": args.job_count,
        "train_snapshots": len(train_snapshots),
        "eval_snapshots": len(eval_snapshots),
        "eval_job_seeds": eval_job_seeds,
        "timesteps": args.timesteps,
        "requested_device": args.device,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "model_path": str(model_path),
        "summary": summary_rows,
    }
    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _evaluate_policy(
    model: DQN,
    snapshots,
    scenario: str,
    job_count: int,
    candidate_count: int,
    job_seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot_index, (timestamp, _) in enumerate(snapshots):
        for job_seed in job_seeds:
            env = CidtTimeseriesDqnEnv(
                snapshots,
                scenario=scenario,
                job_count=job_count,
                candidate_count=candidate_count,
                seed=job_seed,
            )
            observation, _ = env.reset(options={"snapshot_index": snapshot_index, "job_seed": job_seed})
            done = False
            allocations = []
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(int(action))
                allocation = info.get("allocation")
                if allocation is not None:
                    allocations.append(allocation)
                done = terminated or truncated
            summary = summarize(allocations, env.jobs)
            rows.append(
                {
                    "timestamp": timestamp,
                    "job_seed": job_seed,
                    "method": "timeseries_dqn_rl",
                    "jobs_total": summary["jobs_total"],
                    "jobs_assigned": summary["jobs_assigned"],
                    "jobs_failed": summary["jobs_failed"],
                    "assignment_rate": summary["assignment_rate"],
                    "average_score": summary["average_score"],
                    "average_cpu_after_percent": summary["average_cpu_after_percent"],
                    "cpu_after_stddev": summary["cpu_after_stddev"],
                    "deadline_risk_jobs": summary["deadline_risk_jobs"],
                    "servers_used": summary["servers_used"],
                    "racks_used": summary["racks_used"],
                }
            )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "method": "timeseries_dqn_rl",
            "trials": len(rows),
            "assignment_rate_mean": _mean(rows, "assignment_rate"),
            "average_score_mean": _mean(rows, "average_score"),
            "jobs_failed_mean": _mean(rows, "jobs_failed"),
            "cpu_after_stddev_mean": _mean(rows, "cpu_after_stddev"),
            "servers_used_mean": _mean(rows, "servers_used"),
            "racks_used_mean": _mean(rows, "racks_used"),
        }
    ]


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 6) if rows else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
