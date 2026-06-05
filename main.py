from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines import HEURISTICS, allocate_jobs
from src.evaluate import comparison_table, summarize
from src.export_results import export_allocation_plan, export_evaluation_table, export_summary
from src.loaders import load_jobs, load_servers, write_jobs
from src.synthetic_jobs import generate_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIDT baseline workload allocation prototype.")
    parser.add_argument("--servers", default="data/servers_snapshot.csv", help="Prometheus server snapshot CSV/JSON.")
    parser.add_argument("--jobs", default="data/jobs.csv", help="Synthetic or provided jobs CSV.")
    parser.add_argument("--generate-jobs", type=int, default=48, help="Generate this many jobs if --jobs is missing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--results-dir", default="results", help="Output directory.")
    parser.add_argument(
        "--heuristic",
        choices=["all", *HEURISTICS.keys()],
        default="all",
        help="Allocator to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    servers = load_servers(args.servers)
    jobs_path = Path(args.jobs)
    if jobs_path.exists():
        jobs = load_jobs(jobs_path)
    else:
        jobs = generate_jobs(args.generate_jobs, servers, seed=args.seed)
        write_jobs(jobs_path, jobs)

    selected = list(HEURISTICS) if args.heuristic == "all" else [args.heuristic]
    results_dir = Path(args.results_dir)
    summaries: list[dict[str, object]] = []
    allocations_by_heuristic = {}

    for heuristic in selected:
        allocations = allocate_jobs(servers, jobs, heuristic, seed=args.seed)
        allocations_by_heuristic[heuristic] = allocations
        summary = summarize(allocations, jobs)
        summaries.append(summary)
        export_allocation_plan(results_dir / f"allocation_plan_{heuristic}.csv", allocations)

    comparison = comparison_table(summaries)
    best_heuristic = str(comparison[0]["heuristic"]) if comparison else selected[0]
    export_allocation_plan(results_dir / "allocation_plan.csv", allocations_by_heuristic[best_heuristic])
    export_evaluation_table(results_dir / "allocation_evaluation.csv", comparison)

    payload = {
        "server_snapshot": args.servers,
        "jobs": str(jobs_path),
        "heuristics": selected,
        "canonical_allocation_plan": f"allocation_plan.csv uses {best_heuristic}",
        "summaries": summaries,
        "comparison": comparison,
    }
    export_summary(results_dir / "allocation_summary.json", payload)

    best = payload["comparison"][0] if payload["comparison"] else {}
    print(f"Loaded {len(servers)} servers and {len(jobs)} jobs.")
    print(f"Wrote results to {results_dir.resolve()}.")
    if best:
        print(
            "Best heuristic: "
            f"{best['heuristic']} "
            f"assignment_rate={best['assignment_rate']} "
            f"average_score={best['average_score']}"
        )


if __name__ == "__main__":
    main()
