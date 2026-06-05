from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

from src.baselines import HEURISTICS, allocate_jobs
from src.evaluate import summarize
from src.export_results import export_allocation_plan
from src.loaders import _server_from_row
from src.rl_train import evaluate_q_policy, train_q_learning
from src.workload_scenarios import SCENARIOS, generate_scenario_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CIDT experiments over a full Prometheus time series export.")
    parser.add_argument("--timeseries", default="data/server_metrics_timeseries.csv")
    parser.add_argument("--job-count", type=int, default=48)
    parser.add_argument("--job-seeds", default="42,43,44")
    parser.add_argument("--scenario", choices=SCENARIOS, default="normal")
    parser.add_argument("--max-snapshots", type=int, default=0, help="0 means use all timestamps.")
    parser.add_argument("--include-q", action="store_true", help="Include tabular Q-learning on selected timestamps.")
    parser.add_argument("--q-episodes", type=int, default=500)
    parser.add_argument("--include-dqn", action="store_true", help="Include Stable-Baselines3 DQN on selected timestamps.")
    parser.add_argument("--dqn-timesteps", type=int, default=3000)
    parser.add_argument("--results-dir", default="results/timeseries_experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_seeds = [int(seed.strip()) for seed in args.job_seeds.split(",") if seed.strip()]
    results_dir = Path(args.results_dir)
    plans_dir = results_dir / "plans"
    results_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    snapshots = list(_iter_snapshots(Path(args.timeseries)))
    selected = _even_sample(snapshots, args.max_snapshots) if args.max_snapshots else snapshots

    trial_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for index, (timestamp, servers) in enumerate(selected, start=1):
        metadata_rows.append(_snapshot_metadata(timestamp, servers))
        for job_seed in job_seeds:
            jobs = generate_scenario_jobs(args.scenario, args.job_count, servers, seed=job_seed)
            for method in HEURISTICS:
                allocations = allocate_jobs(servers, jobs, method, seed=job_seed)
                trial_rows.append(_trial_row(timestamp, job_seed, summarize(allocations, jobs)))
            if args.include_q:
                q_table, _ = train_q_learning(servers, jobs, episodes=args.q_episodes, seed=job_seed)
                q_allocations, q_summary = evaluate_q_policy(servers, jobs, q_table)
                trial_rows.append(_trial_row(timestamp, job_seed, q_summary))
                export_allocation_plan(plans_dir / f"{_safe_timestamp(timestamp)}_seed_{job_seed}_q_learning_rl.csv", q_allocations)
            if args.include_dqn:
                dqn_allocations, dqn_summary = _run_dqn_trial(servers, jobs, args.dqn_timesteps, job_seed)
                trial_rows.append(_trial_row(timestamp, job_seed, dqn_summary))
                export_allocation_plan(plans_dir / f"{_safe_timestamp(timestamp)}_seed_{job_seed}_dqn_rl.csv", dqn_allocations)
        if index == 1 or index % 100 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} timestamps")

    summary_rows = _aggregate_rows(trial_rows)
    _write_csv(results_dir / "timeseries_trials.csv", trial_rows)
    _write_csv(results_dir / "timeseries_summary.csv", summary_rows)
    _write_csv(results_dir / "timeseries_snapshot_metadata.csv", metadata_rows)
    (results_dir / "timeseries_summary.json").write_text(
        json.dumps(
            {
                "timeseries": args.timeseries,
                "timestamps_total": len(snapshots),
                "timestamps_used": len(selected),
                "job_seeds": job_seeds,
                "scenario": args.scenario,
                "job_count": args.job_count,
                "include_q": args.include_q,
                "include_dqn": args.include_dqn,
                "summary": summary_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Ran {len(trial_rows)} trials over {len(selected)} timestamps.")
    print(f"Wrote outputs to {results_dir.resolve()}.")


def _iter_snapshots(path: Path) -> Iterable[tuple[str, list]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        current_timestamp = ""
        rows: list[dict[str, object]] = []
        for row in reader:
            timestamp = str(row["timestamp"])
            if current_timestamp and timestamp != current_timestamp:
                yield current_timestamp, [_server_from_row(item) for item in rows]
                rows = []
            current_timestamp = timestamp
            rows.append(row)
        if rows:
            yield current_timestamp, [_server_from_row(item) for item in rows]


def _even_sample(snapshots: list[tuple[str, list]], count: int) -> list[tuple[str, list]]:
    if count <= 0 or count >= len(snapshots):
        return snapshots
    if count == 1:
        return [snapshots[0]]
    indexes = sorted({round(index * (len(snapshots) - 1) / (count - 1)) for index in range(count)})
    return [snapshots[index] for index in indexes]


def _snapshot_metadata(timestamp: str, servers: list) -> dict[str, Any]:
    up_servers = [server for server in servers if server.up]
    cpu_values = [server.cpu_usage_percent for server in up_servers if server.cpu_cores > 0]
    mem_values = [
        1.0 - server.mem_available_gb / max(server.mem_total_gb, 1.0)
        for server in up_servers
        if server.mem_total_gb > 0
    ]
    return {
        "timestamp": timestamp,
        "servers_total": len(servers),
        "servers_up": len(up_servers),
        "servers_down": len(servers) - len(up_servers),
        "avg_cpu_usage_percent": round(mean(cpu_values), 6) if cpu_values else 0.0,
        "avg_mem_used_ratio": round(mean(mem_values), 6) if mem_values else 0.0,
    }


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


def _safe_timestamp(timestamp: str) -> str:
    return timestamp.replace(":", "").replace(" ", "_").replace("-", "")


def _trial_row(timestamp: str, job_seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
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
