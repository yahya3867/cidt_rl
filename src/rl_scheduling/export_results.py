from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from .models import Allocation


def export_allocation_plan(path: str | Path, allocations: list[Allocation]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(allocations[0]).keys()) if allocations else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for allocation in allocations:
            writer.writerow(asdict(allocation))


def export_summary(path: str | Path, payload: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_evaluation_table(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "heuristic",
        "jobs_total",
        "jobs_assigned",
        "jobs_failed",
        "assignment_rate",
        "average_score",
        "average_cpu_after_percent",
        "cpu_after_stddev",
        "deadline_risk_jobs",
        "servers_used",
        "racks_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
