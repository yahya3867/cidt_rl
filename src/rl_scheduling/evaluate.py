from __future__ import annotations

from collections import Counter
from statistics import pstdev

from .models import Allocation, Job


def summarize(allocations: list[Allocation], jobs: list[Job]) -> dict[str, object]:
    job_by_id = {job.job_id: job for job in jobs}
    assigned = [item for item in allocations if item.assigned]
    failed = [item for item in allocations if not item.assigned]
    rack_counts = Counter(item.rack for item in assigned)
    server_counts = Counter(item.instance for item in assigned)
    deadline_risk = sum(
        1 for item in assigned if job_by_id[item.job_id].runtime_minutes > job_by_id[item.job_id].deadline_minutes
    )
    cpu_values = [item.cpu_after_percent for item in assigned]

    return {
        "heuristic": allocations[0].heuristic if allocations else "",
        "jobs_total": len(jobs),
        "jobs_assigned": len(assigned),
        "jobs_failed": len(failed),
        "assignment_rate": round(len(assigned) / max(len(jobs), 1), 4),
        "average_score": round(sum(item.score for item in allocations) / max(len(allocations), 1), 4),
        "average_cpu_after_percent": round(sum(cpu_values) / max(len(cpu_values), 1), 4),
        "cpu_after_stddev": round(pstdev(cpu_values), 4) if len(cpu_values) > 1 else 0.0,
        "deadline_risk_jobs": deadline_risk,
        "servers_used": len(server_counts),
        "racks_used": len(rack_counts),
        "top_racks": dict(rack_counts.most_common(5)),
        "failed_reasons": dict(Counter(item.reason for item in failed)),
    }


def comparison_table(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        summaries,
        key=lambda row: (
            float(row["assignment_rate"]),
            float(row["average_score"]),
            -float(row["cpu_after_stddev"]),
        ),
        reverse=True,
    )
