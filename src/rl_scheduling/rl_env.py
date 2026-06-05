from __future__ import annotations

import math

from .baselines import CPU_OVERLOAD_LIMIT, MutableServer
from .models import Allocation, Job, Server


class CidtAllocationEnv:
    """Small Gymnasium-shaped environment without external dependencies."""

    def __init__(self, servers: list[Server], jobs: list[Job]) -> None:
        self.servers = servers
        self.jobs = sorted(jobs, key=lambda item: (-item.priority, item.deadline_minutes, item.job_id))
        self.pool: list[MutableServer] = []
        self.job_index = 0

    @property
    def action_count(self) -> int:
        return len(self.servers)

    def reset(self) -> tuple[tuple[int, ...], dict[str, object]]:
        self.pool = [MutableServer(server) for server in self.servers]
        self.job_index = 0
        return self.state_key(), {"job_id": self.current_job.job_id if self.jobs else ""}

    @property
    def current_job(self) -> Job:
        return self.jobs[self.job_index]

    def valid_actions(self) -> list[int]:
        if self.job_index >= len(self.jobs):
            return []
        job = self.current_job
        return [index for index, server in enumerate(self.pool) if server.can_fit(job)]

    def state_key(self) -> tuple[int, ...]:
        if self.job_index >= len(self.jobs):
            return (0, 0, 0, 0, 0, 0)

        job = self.current_job
        valid_count = len(self.valid_actions())
        active_servers = [server for server in self.pool if server.up and server.cpu_cores > 0]
        avg_cpu = sum(server.cpu_after_percent for server in active_servers) / max(len(active_servers), 1)
        avg_mem = sum(server.mem_available_gb for server in active_servers) / max(len(active_servers), 1)
        avg_disk = sum(server.disk_available_gb for server in active_servers) / max(len(active_servers), 1)

        return (
            _bucket(job.cpu_cores, [2, 8, 16, 32]),
            _bucket(job.memory_gb, [8, 32, 64, 128]),
            _bucket(job.disk_gb, [40, 100, 250, 500]),
            _bucket(valid_count, [1, 5, 15, 40]),
            _bucket(avg_cpu, [20, 40, 60, 80]),
            _bucket(avg_mem + math.log1p(avg_disk), [64, 128, 256, 512]),
        )

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict[str, object]]:
        job = self.current_job
        server = self.pool[action] if 0 <= action < len(self.pool) else None

        if server is None:
            reward = -50.0
            allocation = self._failed_allocation(job, "invalid_action", reward)
        elif not server.up:
            reward = -45.0 - job.priority
            allocation = self._failed_allocation(job, "server_down", reward)
        elif not server.can_fit(job):
            reward = self._resource_penalty(server, job)
            allocation = self._failed_allocation(job, "resource_violation", reward)
        else:
            server.place(job)
            reward = self._placement_reward(server, job)
            allocation = Allocation(
                heuristic="q_learning_rl",
                job_id=job.job_id,
                assigned=True,
                instance=server.instance,
                rack=server.rack,
                unit=server.unit,
                reason="placed",
                cpu_after_percent=round(server.cpu_after_percent, 3),
                mem_remaining_gb=round(server.mem_available_gb, 3),
                disk_remaining_gb=round(server.disk_available_gb, 3),
                score=round(reward, 3),
            )

        self.job_index += 1
        done = self.job_index >= len(self.jobs)
        return self.state_key(), reward, done, {"allocation": allocation}

    def _placement_reward(self, server: MutableServer, job: Job) -> float:
        cpu_util = server.cpu_after_percent / 100.0
        target_utilization_bonus = 4.0 - abs(0.70 - cpu_util) * 4.0
        deadline_bonus = 2.0 * max(0.0, 1.0 - job.runtime_minutes / max(job.deadline_minutes, 1))
        rack_bonus = 1.5 if job.preferred_rack and job.preferred_rack == server.rack else 0.0
        pressure_penalty = (job.memory_gb / max(server.mem_available_gb + job.memory_gb, 1.0)) * 2.0
        return 10.0 + job.priority + target_utilization_bonus + deadline_bonus + rack_bonus - pressure_penalty

    def _resource_penalty(self, server: MutableServer, job: Job) -> float:
        if server.cpu_cores <= 0:
            return -35.0 - job.priority
        projected_cpu = server.cpu_after_percent + (100.0 * job.cpu_cores / server.cpu_cores)
        if projected_cpu > CPU_OVERLOAD_LIMIT:
            return -30.0 - job.priority
        if server.mem_available_gb < job.memory_gb:
            return -25.0 - job.priority
        if server.disk_available_gb < job.disk_gb:
            return -20.0 - job.priority
        return -15.0 - job.priority

    @staticmethod
    def _failed_allocation(job: Job, reason: str, reward: float) -> Allocation:
        return Allocation(
            heuristic="q_learning_rl",
            job_id=job.job_id,
            assigned=False,
            instance="",
            rack="",
            unit="",
            reason=reason,
            cpu_after_percent=0.0,
            mem_remaining_gb=0.0,
            disk_remaining_gb=0.0,
            score=round(reward, 3),
        )


def _bucket(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)
