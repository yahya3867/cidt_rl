from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .models import Job, Server


def _as_float(value: object, default: float = 0.0) -> float:
    if value in (None, "", "None", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    return int(round(_as_float(value, default)))


def _server_from_row(row: dict[str, object]) -> Server:
    cpu_cores = _as_float(row.get("cpu_cores"))
    mem_total = _as_float(row.get("mem_total_gb"))
    disk_size = _as_float(row.get("disk_size_gb"))
    up = bool(_as_int(row.get("up")))

    return Server(
        timestamp=str(row.get("timestamp", "")),
        instance=str(row.get("instance", "")),
        ip=str(row.get("ip", "")),
        rack=str(row.get("rack", "")),
        unit=str(row.get("unit", "")),
        note=str(row.get("note", "")),
        user=str(row.get("user", "")),
        up=up,
        cpu_usage_percent=_as_float(row.get("cpu_usage_percent"), 100.0 if not up else 0.0),
        cpu_cores=cpu_cores,
        mem_available_gb=_as_float(row.get("mem_available_gb")),
        mem_total_gb=mem_total,
        disk_available_gb=_as_float(row.get("disk_available_gb")),
        disk_size_gb=disk_size,
        load1=_as_float(row.get("load1")),
        load5=_as_float(row.get("load5")),
        load15=_as_float(row.get("load15")),
    )


def load_servers(path: str | Path) -> list[Server]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [_server_from_row(row) for row in rows]

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_server_from_row(row) for row in csv.DictReader(handle)]


def load_jobs(path: str | Path) -> list[Job]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            Job(
                job_id=str(row["job_id"]),
                cpu_cores=_as_float(row["cpu_cores"]),
                memory_gb=_as_float(row["memory_gb"]),
                disk_gb=_as_float(row["disk_gb"]),
                runtime_minutes=_as_int(row["runtime_minutes"]),
                deadline_minutes=_as_int(row["deadline_minutes"]),
                priority=_as_int(row["priority"], 1),
                preferred_rack=str(row.get("preferred_rack", "")),
            )
            for row in rows
        ]


def write_jobs(path: str | Path, jobs: Iterable[Job]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "job_id",
        "cpu_cores",
        "memory_gb",
        "disk_gb",
        "runtime_minutes",
        "deadline_minutes",
        "priority",
        "preferred_rack",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({field: getattr(job, field) for field in fields})
