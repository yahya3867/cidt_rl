from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("results")


def main() -> None:
    sections = [
        "# CIDT Scheduling Experiment Results",
        "",
        "This report consolidates the CIDT workload-placement experiments run so far. Higher `average_score_mean` is better. Higher `assignment_rate_mean` is better. Lower `jobs_failed_mean` is better.",
        "",
        "## Data Used",
        "",
        "| Dataset | Scope | Notes |",
        "| --- | ---: | --- |",
        "| `data/server_metrics_timeseries.csv` | 242,352 rows | Full Prometheus export |",
        "| Time span | 2026-04-30 07:00 to 2026-05-14 15:00 | 1,377 timestamps |",
        "| Servers per timestamp | 176 | Same schema across the full export |",
        "",
        "## Algorithms Compared",
        "",
        "| Method | Type | What it does |",
        "| --- | --- | --- |",
        "| `random_valid_server` | Heuristic | Randomly selects any up server with enough CPU, memory, and disk. |",
        "| `least_loaded_server` | Heuristic | Selects the valid server with lowest CPU usage. |",
        "| `most_available_memory` | Heuristic | Selects the valid server with most available memory. |",
        "| `best_fit_resource_match` | Heuristic | Selects the valid server whose remaining resources best match the job. |",
        "| `rack_aware_placement` | Heuristic | Prefers the job's rack when possible, then uses least-loaded behavior. |",
        "| `failure_aware_placement` | Heuristic | Avoids down servers and scores resource fit, priority, utilization, and deadline slack. |",
        "| `q_learning_rl` | RL | Dependency-free tabular Q-learning scheduler. |",
        "| `dqn_rl` | RL | Gymnasium + Stable-Baselines3 DQN scheduler. |",
        "",
        "## 1. Full Normal Time-Series Baseline Results",
        "",
        "Run over the complete Prometheus time series using normal synthetic workloads.",
        "",
        _table(ROOT / "timeseries_experiments" / "timeseries_summary.csv"),
        "",
        "**Interpretation:** Under normal synthetic workloads, all heuristic methods place all jobs. `failure_aware_placement` has the best mean score.",
        "",
        "## 2. Full Time-Series RL Sample",
        "",
        "Run on 12 representative timestamps from the full time series, including tabular Q-learning and DQN.",
        "",
        _table(ROOT / "timeseries_experiments_rl_sample" / "timeseries_summary.csv"),
        "",
        "**Interpretation:** DQN places all jobs, but it does not beat the strongest heuristic. RL is feasible, but not yet superior.",
        "",
        "## 3. Stress Scenario Results",
        "",
        "Each stress scenario used 100 representative timestamps, 3 job seeds, and 200 jobs per trial.",
        "",
        _table(ROOT / "stress_scenario_summary.csv", group_by="scenario"),
        "",
        "## 4. Best Method By Scenario",
        "",
        _best_by_group_table(ROOT / "stress_scenario_summary.csv", "scenario"),
        "",
        "## 5. Earlier Pilot Snapshot Experiments",
        "",
        "These were smaller experiments over six individual snapshot CSVs. They are included for completeness, but the full time-series results above should be treated as the main results.",
        "",
        "### Six-Snapshot Baselines + Q-Learning",
        "",
        _table(ROOT / "experiments" / "experiment_summary.csv"),
        "",
        "### Six-Snapshot DQN-Inclusive Pilot",
        "",
        _table(ROOT / "experiments_dqn" / "experiment_summary.csv"),
        "",
        "## Overall Conclusion",
        "",
        "- In normal/easy workloads, all methods achieve complete placement; `failure_aware_placement` is the strongest.",
        "- Under stress, no method places all jobs. This is good experimentally because the problem is now hard enough to compare algorithms.",
        "- Under `mixed_stress`, `random_valid_server` and `best_fit_resource_match` outperform the conservative failure-aware heuristic.",
        "- Under `disk_pressure`, `rack_aware_placement` and `least_loaded_server` are strongest.",
        "- Under `memory_pressure`, all methods struggle; this is the hardest scenario.",
        "- Current RL methods are valid and learn feasible allocation behavior, but they do not yet outperform the best heuristics.",
        "",
        "## Research Takeaway",
        "",
        "The current CIDT experiments reproduce the scheduling-comparison structure of the papers: heuristics and RL are evaluated on the same environment and metrics. The current result is not that RL wins. The current result is that heuristic scheduling remains stronger for this CIDT prototype, especially before the RL reward/state design is tuned further.",
        "",
    ]

    Path("result.md").write_text("\n".join(sections), encoding="utf-8")
    print("Wrote result.md")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _table(path: Path, group_by: str | None = None) -> str:
    rows = _read_rows(path)
    wanted = [
        column
        for column in [
            group_by,
            "method",
            "trials",
            "assignment_rate_mean",
            "average_score_mean",
            "average_score_std",
            "jobs_failed_mean",
            "cpu_after_stddev_mean",
            "servers_used_mean",
            "racks_used_mean",
        ]
        if column and column in rows[0]
    ]
    lines = [_markdown_header(wanted)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row[column]) for column in wanted) + " |")
    return "\n".join(lines)


def _best_by_group_table(path: Path, group_column: str) -> str:
    rows = _read_rows(path)
    groups = sorted({row[group_column] for row in rows})
    best_rows = []
    for group in groups:
        group_rows = [row for row in rows if row[group_column] == group]
        best = sorted(
            group_rows,
            key=lambda row: (float(row["assignment_rate_mean"]), float(row["average_score_mean"])),
            reverse=True,
        )[0]
        best_rows.append(best)
    wanted = [group_column, "method", "assignment_rate_mean", "average_score_mean", "jobs_failed_mean"]
    lines = [_markdown_header(wanted)]
    for row in best_rows:
        lines.append("| " + " | ".join(_fmt(row[column]) for column in wanted) + " |")
    return "\n".join(lines)


def _markdown_header(columns: list[str]) -> str:
    return "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |"


def _fmt(value: str) -> str:
    try:
        numeric = float(value)
    except ValueError:
        return value
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.4f}"


if __name__ == "__main__":
    main()
