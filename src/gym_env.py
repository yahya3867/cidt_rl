from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .baselines import CPU_OVERLOAD_LIMIT, MutableServer
from .models import Allocation, Job, Server


class CidtGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        servers: list[Server],
        jobs: list[Job],
        invalid_action_penalty: float = -35.0,
        candidate_count: int = 16,
    ) -> None:
        super().__init__()
        self.servers = servers
        self.jobs = sorted(jobs, key=lambda item: (-item.priority, item.deadline_minutes, item.job_id))
        self.invalid_action_penalty = invalid_action_penalty
        self.candidate_count = candidate_count
        self.pool: list[MutableServer] = []
        self.job_index = 0
        self.last_allocation: Allocation | None = None

        self.action_space = spaces.Discrete(self.candidate_count)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.pool = [MutableServer(server) for server in self.servers]
        self.job_index = 0
        self.last_allocation = None
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        job = self.jobs[self.job_index]
        candidates = self._candidate_servers(job)
        server = candidates[int(action)] if 0 <= int(action) < len(candidates) else None

        if server is None:
            reward = self.invalid_action_penalty
            allocation = self._failed_allocation(job, "invalid_candidate_action", reward)
        elif not server.up:
            reward = self.invalid_action_penalty - job.priority
            allocation = self._failed_allocation(job, "server_down", reward)
        elif not server.can_fit(job):
            reward = self._resource_penalty(server, job)
            allocation = self._failed_allocation(job, "resource_violation", reward)
        else:
            server.place(job)
            reward = self._placement_reward(server, job)
            allocation = Allocation(
                heuristic="dqn_rl",
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

        self.last_allocation = allocation
        self.job_index += 1
        terminated = self.job_index >= len(self.jobs)
        return self._observation(), float(reward), terminated, False, self._info(allocation)

    def action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        if self.job_index >= len(self.jobs):
            return mask
        for index, _ in enumerate(self._candidate_servers(self.jobs[self.job_index])):
            mask[index] = 1
        return mask

    def _observation(self) -> np.ndarray:
        if self.job_index >= len(self.jobs):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        job = self.jobs[self.job_index]
        active = [server for server in self.pool if server.up and server.cpu_cores > 0]
        valid = [server for server in active if server.can_fit(job)]
        max_cpu = max((server.cpu_cores for server in active), default=1.0)
        max_mem = max((server.mem_available_gb for server in active), default=1.0)
        max_disk = max((server.disk_available_gb for server in active), default=1.0)
        avg_cpu = sum(server.cpu_after_percent for server in active) / max(len(active), 1)
        avg_mem_ratio = sum(server.mem_available_gb / max(server.original.mem_total_gb, 1.0) for server in active) / max(len(active), 1)
        avg_disk_ratio = sum(server.disk_available_gb / max(server.original.disk_size_gb, 1.0) for server in active) / max(len(active), 1)
        max_fit_cpu = max(((server.cpu_cores - server.cpu_used) / max(job.cpu_cores, 1.0) for server in valid), default=0.0)
        max_fit_mem = max((server.mem_available_gb / max(job.memory_gb, 1.0) for server in valid), default=0.0)
        max_fit_disk = max((server.disk_available_gb / max(job.disk_gb, 1.0) for server in valid), default=0.0)

        return np.array(
            [
                min(job.cpu_cores / max_cpu, 1.0),
                min(job.memory_gb / max_mem, 1.0),
                min(job.disk_gb / max_disk, 1.0),
                min(job.runtime_minutes / 240.0, 1.0),
                min(job.deadline_minutes / 480.0, 1.0),
                job.priority / 5.0,
                min(len(valid) / max(self.candidate_count, 1), 1.0),
                len(active) / max(len(self.pool), 1),
                min(avg_cpu / 100.0, 1.0),
                min(max(avg_mem_ratio, 0.0), 1.0),
                min(max(avg_disk_ratio, 0.0), 1.0),
                min((max_fit_cpu + max_fit_mem + max_fit_disk) / 30.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _info(self, allocation: Allocation | None = None) -> dict[str, Any]:
        return {
            "job_index": self.job_index,
            "job_id": self.jobs[self.job_index].job_id if self.job_index < len(self.jobs) else "",
            "valid_actions": int(self.action_mask().sum()),
            "allocation": allocation,
        }

    def _candidate_servers(self, job: Job) -> list[MutableServer]:
        valid = [server for server in self.pool if server.can_fit(job)]
        ranked = sorted(
            valid,
            key=lambda server: (
                abs(0.70 - ((server.cpu_used + job.cpu_cores) / max(server.cpu_cores, 1.0))),
                abs(server.mem_available_gb - job.memory_gb),
                abs(server.disk_available_gb - job.disk_gb),
            ),
        )
        return ranked[: self.candidate_count]

    def _placement_reward(self, server: MutableServer, job: Job) -> float:
        cpu_util = server.cpu_after_percent / 100.0
        utilization_bonus = 5.0 - abs(0.70 - cpu_util) * 5.0
        deadline_bonus = 2.0 * max(0.0, 1.0 - job.runtime_minutes / max(job.deadline_minutes, 1))
        rack_bonus = 1.0 if job.preferred_rack and job.preferred_rack == server.rack else 0.0
        imbalance_penalty = abs(cpu_util - 0.70) * 2.0
        return 10.0 + job.priority + utilization_bonus + deadline_bonus + rack_bonus - imbalance_penalty

    def _resource_penalty(self, server: MutableServer, job: Job) -> float:
        if server.cpu_cores <= 0:
            return self.invalid_action_penalty - job.priority
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
            heuristic="dqn_rl",
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
