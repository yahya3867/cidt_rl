from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.export_results import export_allocation_plan, export_summary
from src.loaders import load_jobs, load_servers
from src.rl_train import evaluate_q_policy, serializable_q_table, train_q_learning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a pure-Python Q-learning CIDT allocator.")
    parser.add_argument("--servers", default="data/servers_snapshot.csv", help="Prometheus server snapshot CSV/JSON.")
    parser.add_argument("--jobs", default="data/jobs.csv", help="Jobs CSV.")
    parser.add_argument("--episodes", type=int, default=800, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    servers = load_servers(args.servers)
    jobs = load_jobs(args.jobs)
    results_dir = Path(args.results_dir)

    q_table, training_metrics = train_q_learning(
        servers=servers,
        jobs=jobs,
        episodes=args.episodes,
        seed=args.seed,
    )
    allocations, summary = evaluate_q_policy(servers, jobs, q_table)

    export_allocation_plan(results_dir / "allocation_plan_q_learning_rl.csv", allocations)
    export_summary(
        results_dir / "rl_training_summary.json",
        {
            "algorithm": "tabular_q_learning_with_action_masking",
            "episodes": args.episodes,
            "seed": args.seed,
            "server_snapshot": args.servers,
            "jobs": args.jobs,
            "evaluation_summary": summary,
            "training_metrics": training_metrics,
        },
    )
    (results_dir / "q_table.json").write_text(
        json.dumps(serializable_q_table(q_table), indent=2),
        encoding="utf-8",
    )

    print(f"Trained Q-learning allocator for {args.episodes} episodes.")
    print(
        "Evaluation: "
        f"assignment_rate={summary['assignment_rate']} "
        f"average_score={summary['average_score']} "
        f"jobs_failed={summary['jobs_failed']}"
    )
    print(f"Wrote RL outputs to {results_dir.resolve()}.")


if __name__ == "__main__":
    main()
