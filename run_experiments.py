from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.baselines import HEURISTICS, allocate_jobs
from src.evaluate import summarize
from src.export_results import export_allocation_plan
from src.loaders import load_servers
from src.rl_train import evaluate_q_policy, train_q_learning
from src.synthetic_jobs import generate_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-snapshot CIDT scheduling experiments.")
    parser.add_argument("--snapshot-glob", default="snapshot_*.csv", help="Glob for Prometheus snapshots.")
    parser.add_argument("--job-count", type=int, default=48, help="Synthetic jobs per trial.")
    parser.add_argument("--job-seeds", default="42,43,44,45,46", help="Comma-separated synthetic job seeds.")
    parser.add_argument("--q-episodes", type=int, default=500, help="Tabular Q-learning episodes per trial.")
    parser.add_argument("--include-dqn", action="store_true", help="Also train/evaluate Stable-Baselines3 DQN.")
    parser.add_argument("--dqn-timesteps", type=int, default=5000, help="DQN timesteps per trial when enabled.")
    parser.add_argument("--results-dir", default="results/experiments", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_paths = sorted(Path(".").glob(args.snapshot_glob))
    if not snapshot_paths:
        raise FileNotFoundError(f"No snapshots matched {args.snapshot_glob!r}")

    job_seeds = [int(seed.strip()) for seed in args.job_seeds.split(",") if seed.strip()]
    results_dir = Path(args.results_dir)
    plans_dir = results_dir / "plans"
    results_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for snapshot_path in snapshot_paths:
        servers = load_servers(snapshot_path)
        for job_seed in job_seeds:
            jobs = generate_jobs(args.job_count, servers, seed=job_seed)
            trial_id = f"{snapshot_path.stem}_seed_{job_seed}"

            for heuristic in HEURISTICS:
                allocations = allocate_jobs(servers, jobs, heuristic, seed=job_seed)
                rows.append(_trial_row(snapshot_path, job_seed, summarize(allocations, jobs)))

            q_table, _ = train_q_learning(servers, jobs, episodes=args.q_episodes, seed=job_seed)
            q_allocations, q_summary = evaluate_q_policy(servers, jobs, q_table)
            rows.append(_trial_row(snapshot_path, job_seed, q_summary))
            export_allocation_plan(plans_dir / f"{trial_id}_q_learning_rl.csv", q_allocations)

            if args.include_dqn:
                dqn_allocations, dqn_summary = _run_dqn_trial(servers, jobs, args.dqn_timesteps, job_seed)
                rows.append(_trial_row(snapshot_path, job_seed, dqn_summary))
                export_allocation_plan(plans_dir / f"{trial_id}_dqn_rl.csv", dqn_allocations)

            print(f"Completed {trial_id}")

    aggregate = _aggregate_rows(rows)
    _write_csv(results_dir / "experiment_trials.csv", rows)
    _write_csv(results_dir / "experiment_summary.csv", aggregate)
    (results_dir / "experiment_summary.json").write_text(
        json.dumps({"trials": rows, "summary": aggregate}, indent=2),
        encoding="utf-8",
    )

    print(f"Ran {len(rows)} method trials over {len(snapshot_paths)} snapshots and {len(job_seeds)} job seeds.")
    print(f"Wrote experiment outputs to {results_dir.resolve()}.")


def _run_dqn_trial(servers, jobs, timesteps: int, seed: int):
    from stable_baselines3 import DQN
    from stable_baselines3.common.monitor import Monitor

    from src.gym_env import CidtGymEnv

    env = Monitor(CidtGymEnv(servers, jobs))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        buffer_size=50000,
        learning_starts=300,
        batch_size=64,
        gamma=0.96,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.40,
        exploration_initial_eps=0.90,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=seed,
        device="auto",
    )
    model.learn(total_timesteps=timesteps, progress_bar=False)
    allocations = _evaluate_dqn_policy(model, CidtGymEnv(servers, jobs))
    return allocations, summarize(allocations, jobs)


def _evaluate_dqn_policy(model, env):
    observations, _ = env.reset()
    done = False
    allocations = []
    while not done:
        action, _ = model.predict(observations, deterministic=True)
        observations, _, terminated, truncated, info = env.step(int(action))
        allocation = info.get("allocation")
        if allocation is not None:
            allocations.append(allocation)
        done = terminated or truncated
    return allocations


def _trial_row(snapshot_path: Path, job_seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot": snapshot_path.name,
        "job_seed": job_seed,
        "method": summary["heuristic"],
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


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
    aggregate: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        aggregate.append(
            {
                "method": method,
                "trials": len(method_rows),
                "assignment_rate_mean": _mean(method_rows, "assignment_rate"),
                "assignment_rate_std": _std(method_rows, "assignment_rate"),
                "average_score_mean": _mean(method_rows, "average_score"),
                "average_score_std": _std(method_rows, "average_score"),
                "jobs_failed_mean": _mean(method_rows, "jobs_failed"),
                "cpu_after_stddev_mean": _mean(method_rows, "cpu_after_stddev"),
                "servers_used_mean": _mean(method_rows, "servers_used"),
                "racks_used_mean": _mean(method_rows, "racks_used"),
            }
        )
    return sorted(aggregate, key=lambda row: (row["assignment_rate_mean"], row["average_score_mean"]), reverse=True)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return round(mean(float(row[key]) for row in rows), 6)


def _std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return round(stdev(values), 6) if len(values) > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
