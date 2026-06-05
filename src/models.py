from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Server:
    timestamp: str
    instance: str
    ip: str
    rack: str
    unit: str
    note: str
    user: str
    up: bool
    cpu_usage_percent: float
    cpu_cores: float
    mem_available_gb: float
    mem_total_gb: float
    disk_available_gb: float
    disk_size_gb: float
    load1: float
    load5: float
    load15: float


@dataclass(frozen=True)
class Job:
    job_id: str
    cpu_cores: float
    memory_gb: float
    disk_gb: float
    runtime_minutes: int
    deadline_minutes: int
    priority: int
    preferred_rack: str = ""


@dataclass(frozen=True)
class Allocation:
    heuristic: str
    job_id: str
    assigned: bool
    instance: str
    rack: str
    unit: str
    reason: str
    cpu_after_percent: float
    mem_remaining_gb: float
    disk_remaining_gb: float
    score: float
