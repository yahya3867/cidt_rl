from __future__ import annotations

import random

from .models import Job, Server
from .synthetic_jobs import generate_jobs

SCENARIOS = [
    "normal",
    "cpu_pressure",
    "memory_pressure",
    "disk_pressure",
    "rack_pressure",
    "mixed_stress",
]


def generate_scenario_jobs(
    scenario: str,
    count: int,
    servers: list[Server],
    seed: int = 42,
) -> list[Job]:
    if scenario == "normal":
        return generate_jobs(count, servers, seed=seed)
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown workload scenario: {scenario}")

    rng = random.Random(seed)
    racks = _active_racks(servers)
    constrained_racks = _constrained_racks(servers)
    jobs: list[Job] = []

    for index in range(1, count + 1):
        if scenario == "cpu_pressure":
            cpu = rng.choice([16, 24, 32, 40, 48])
            memory = rng.choice([16, 32, 48, 64])
            disk = rng.choice([50, 100, 200])
            preferred_rack = rng.choice(racks) if rng.random() < 0.35 else ""
        elif scenario == "memory_pressure":
            cpu = rng.choice([4, 8, 12, 16])
            memory = rng.choice([64, 96, 128, 192, 240])
            disk = rng.choice([50, 100, 200])
            preferred_rack = rng.choice(racks) if rng.random() < 0.35 else ""
        elif scenario == "disk_pressure":
            cpu = rng.choice([4, 8, 12, 16])
            memory = rng.choice([16, 32, 64])
            disk = rng.choice([500, 1000, 2000, 4000, 8000])
            preferred_rack = rng.choice(racks) if rng.random() < 0.35 else ""
        elif scenario == "rack_pressure":
            cpu = rng.choice([8, 16, 24, 32])
            memory = rng.choice([32, 64, 96, 128])
            disk = rng.choice([100, 250, 500])
            preferred_rack = rng.choice(constrained_racks)
        else:
            profile = rng.choices(
                ["cpu", "memory", "disk", "large"],
                weights=[0.32, 0.30, 0.18, 0.20],
                k=1,
            )[0]
            if profile == "cpu":
                cpu, memory, disk = rng.choice([24, 32, 40, 48]), rng.choice([24, 48, 64]), rng.choice([100, 250])
            elif profile == "memory":
                cpu, memory, disk = rng.choice([8, 12, 16]), rng.choice([96, 128, 192, 240]), rng.choice([100, 250])
            elif profile == "disk":
                cpu, memory, disk = rng.choice([8, 16]), rng.choice([32, 64, 96]), rng.choice([1000, 2000, 4000])
            else:
                cpu, memory, disk = rng.choice([16, 24, 32]), rng.choice([64, 96, 128]), rng.choice([250, 500, 1000])
            preferred_rack = rng.choice(constrained_racks if rng.random() < 0.60 else racks)

        runtime = rng.choice([30, 60, 90, 120, 180, 240])
        slack = rng.choice([15, 30, 60, 120])
        jobs.append(
            Job(
                job_id=f"{scenario}_job_{index:03d}",
                cpu_cores=float(cpu),
                memory_gb=float(memory),
                disk_gb=float(disk),
                runtime_minutes=runtime,
                deadline_minutes=runtime + slack,
                priority=rng.randint(1, 5),
                preferred_rack=preferred_rack,
            )
        )

    return jobs


def _active_racks(servers: list[Server]) -> list[str]:
    racks = sorted({server.rack for server in servers if server.up and server.rack})
    return racks or sorted({server.rack for server in servers if server.rack}) or [""]


def _constrained_racks(servers: list[Server]) -> list[str]:
    rack_capacity: dict[str, float] = {}
    for server in servers:
        if not server.up or not server.rack:
            continue
        rack_capacity[server.rack] = rack_capacity.get(server.rack, 0.0) + server.cpu_cores + (server.mem_available_gb / 8.0)
    if not rack_capacity:
        return _active_racks(servers)
    ranked = sorted(rack_capacity, key=rack_capacity.get)
    return ranked[: max(1, min(2, len(ranked)))]
