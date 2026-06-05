from __future__ import annotations

import random
from collections.abc import Callable

from .models import Allocation, Job, Server

CPU_OVERLOAD_LIMIT = 95.0


class MutableServer:
    def __init__(self, server: Server) -> None:
        self.original = server
        self.instance = server.instance
        self.rack = server.rack
        self.unit = server.unit
        self.up = server.up
        self.cpu_cores = server.cpu_cores
        self.cpu_used = server.cpu_cores * server.cpu_usage_percent / 100.0
        self.mem_available_gb = server.mem_available_gb
        self.disk_available_gb = server.disk_available_gb

    @property
    def cpu_after_percent(self) -> float:
        if self.cpu_cores <= 0:
            return 100.0
        return 100.0 * self.cpu_used / self.cpu_cores

    def can_fit(self, job: Job, avoid_down: bool = True) -> bool:
        if avoid_down and not self.up:
            return False
        if self.cpu_cores <= 0:
            return False
        if self.cpu_after_percent + (100.0 * job.cpu_cores / self.cpu_cores) > CPU_OVERLOAD_LIMIT:
            return False
        return (
            self.mem_available_gb >= job.memory_gb
            and self.disk_available_gb >= job.disk_gb
        )

    def place(self, job: Job) -> None:
        self.cpu_used += job.cpu_cores
        self.mem_available_gb -= job.memory_gb
        self.disk_available_gb -= job.disk_gb


def _valid_servers(pool: list[MutableServer], job: Job, avoid_down: bool = True) -> list[MutableServer]:
    return [server for server in pool if server.can_fit(job, avoid_down=avoid_down)]


def _failure_reason(pool: list[MutableServer], job: Job) -> str:
    if not any(server.up for server in pool):
        return "all_servers_down"
    up_servers = [server for server in pool if server.up]
    if not any(server.cpu_cores > 0 for server in up_servers):
        return "missing_cpu_capacity"
    if not any(server.cpu_after_percent + (100.0 * job.cpu_cores / server.cpu_cores) <= CPU_OVERLOAD_LIMIT for server in up_servers if server.cpu_cores > 0):
        return "insufficient_cpu_capacity"
    if not any(server.mem_available_gb >= job.memory_gb for server in up_servers):
        return "insufficient_memory"
    if not any(server.disk_available_gb >= job.disk_gb for server in up_servers):
        return "insufficient_disk"
    return "no_valid_server"


def _score(server: MutableServer, job: Job) -> float:
    if not server.up:
        return -100.0
    cpu_util = (server.cpu_used + job.cpu_cores) / server.cpu_cores if server.cpu_cores else 1.0
    mem_pressure = job.memory_gb / max(server.mem_available_gb + job.memory_gb, 1.0)
    disk_pressure = job.disk_gb / max(server.disk_available_gb + job.disk_gb, 1.0)
    deadline_bonus = max(0.0, 1.0 - job.runtime_minutes / max(job.deadline_minutes, 1))
    return 10.0 + 3.0 * job.priority + deadline_bonus - abs(0.70 - cpu_util) - mem_pressure - disk_pressure


def random_valid(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job)
    return rng.choice(candidates) if candidates else None


def least_loaded(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job)
    return min(candidates, key=lambda server: (server.cpu_after_percent, -server.mem_available_gb), default=None)


def most_available_memory(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job)
    return max(candidates, key=lambda server: (server.mem_available_gb, server.disk_available_gb), default=None)


def best_fit_resource_match(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job)
    return min(
        candidates,
        key=lambda server: (
            abs((server.cpu_cores - server.cpu_used) - job.cpu_cores),
            abs(server.mem_available_gb - job.memory_gb),
            abs(server.disk_available_gb - job.disk_gb),
        ),
        default=None,
    )


def rack_aware(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job)
    if job.preferred_rack:
        rack_matches = [server for server in candidates if server.rack == job.preferred_rack]
        if rack_matches:
            return min(rack_matches, key=lambda server: (server.cpu_after_percent, -server.mem_available_gb))
    return least_loaded(job, pool, rng)


def failure_aware(job: Job, pool: list[MutableServer], rng: random.Random) -> MutableServer | None:
    candidates = _valid_servers(pool, job, avoid_down=True)
    return max(candidates, key=lambda server: _score(server, job), default=None)


HEURISTICS: dict[str, Callable[[Job, list[MutableServer], random.Random], MutableServer | None]] = {
    "random_valid_server": random_valid,
    "least_loaded_server": least_loaded,
    "most_available_memory": most_available_memory,
    "best_fit_resource_match": best_fit_resource_match,
    "rack_aware_placement": rack_aware,
    "failure_aware_placement": failure_aware,
}


def allocate_jobs(
    servers: list[Server],
    jobs: list[Job],
    heuristic_name: str,
    seed: int = 42,
) -> list[Allocation]:
    if heuristic_name not in HEURISTICS:
        raise ValueError(f"Unknown heuristic: {heuristic_name}")

    rng = random.Random(seed)
    choose = HEURISTICS[heuristic_name]
    pool = [MutableServer(server) for server in servers]
    allocations: list[Allocation] = []

    for job in sorted(jobs, key=lambda item: (-item.priority, item.deadline_minutes, item.job_id)):
        server = choose(job, pool, rng)
        if server is None:
            allocations.append(
                Allocation(
                    heuristic=heuristic_name,
                    job_id=job.job_id,
                    assigned=False,
                    instance="",
                    rack="",
                    unit="",
                    reason=_failure_reason(pool, job),
                    cpu_after_percent=0.0,
                    mem_remaining_gb=0.0,
                    disk_remaining_gb=0.0,
                    score=-25.0 - job.priority,
                )
            )
            continue

        server.place(job)
        allocations.append(
            Allocation(
                heuristic=heuristic_name,
                job_id=job.job_id,
                assigned=True,
                instance=server.instance,
                rack=server.rack,
                unit=server.unit,
                reason="placed",
                cpu_after_percent=round(server.cpu_after_percent, 3),
                mem_remaining_gb=round(server.mem_available_gb, 3),
                disk_remaining_gb=round(server.disk_available_gb, 3),
                score=round(_score(server, job), 3),
            )
        )

    return allocations
