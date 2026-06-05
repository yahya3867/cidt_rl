from __future__ import annotations

import random

from .models import Job, Server


def generate_jobs(count: int, servers: list[Server], seed: int = 42) -> list[Job]:
    rng = random.Random(seed)
    active_racks = sorted({server.rack for server in servers if server.up and server.rack})
    racks = active_racks or sorted({server.rack for server in servers if server.rack}) or [""]

    jobs: list[Job] = []
    for index in range(1, count + 1):
        profile = rng.choices(
            population=["small", "medium", "large"],
            weights=[0.50, 0.32, 0.18],
            k=1,
        )[0]
        if profile == "small":
            cpu, memory, disk = rng.choice([1, 2, 4]), rng.choice([2, 4, 8]), rng.choice([10, 20, 40])
        elif profile == "medium":
            cpu, memory, disk = rng.choice([4, 8, 12]), rng.choice([16, 24, 32]), rng.choice([50, 100, 150])
        else:
            cpu, memory, disk = rng.choice([16, 24, 32]), rng.choice([48, 64, 96]), rng.choice([150, 250, 400])

        runtime = rng.choice([15, 30, 45, 60, 90, 120, 180])
        slack = rng.choice([30, 60, 120, 240])
        jobs.append(
            Job(
                job_id=f"job_{index:03d}",
                cpu_cores=float(cpu),
                memory_gb=float(memory),
                disk_gb=float(disk),
                runtime_minutes=runtime,
                deadline_minutes=runtime + slack,
                priority=rng.randint(1, 5),
                preferred_rack=rng.choice(racks) if rng.random() < 0.55 else "",
            )
        )

    return jobs
