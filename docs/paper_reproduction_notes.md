# Paper-to-CIDT Reproduction Notes

The six provided papers are treated as design inspiration for the first reproducible coding milestone, not as a full algorithmic replication yet. The prototype reproduces their common scheduling structure in a data-center setting:

- A machine/resource table becomes the Prometheus server snapshot.
- Production jobs become synthetic workloads with CPU, memory, disk, runtime, deadline, and priority.
- Dispatching rules become baseline allocators.
- Dynamic/failure-aware scheduling becomes filtering out down servers and recomputing capacity after each placement.
- RL scheduling becomes the next step once baselines and metrics are stable.

## Current Reproduction Layer

| Paper theme | CIDT prototype equivalent |
| --- | --- |
| Job-shop and flow-shop scheduling | Sequential workload placement into server machines |
| Due-date constrained scheduling | Jobs include runtime and deadline fields |
| Discrete-event scheduling framework | `main.py` runs a deterministic scheduling episode over a job queue |
| RL scheduler for HPC/data-center workloads | Baseline methods and state/action/reward shape are ready for a Gymnasium environment |
| Sim2Real gap reduction | Real Prometheus snapshot plus synthetic workload stress cases |
| Collaborative project-management practices | Modular files and reproducible CSV/JSON outputs |

## Next Reproduction Layer

1. Replace the dependency-free `CidtAllocationEnv` with a formal Gymnasium environment when dependencies are available.
2. Use the same `Server` and `Job` tables as observations.
3. Keep the action space as selecting one server for the current job.
4. Extend rewards for valid placement, utilization, balance, deadlines, and failure avoidance.
5. Compare RL policies against the six baseline heuristic outputs in `results/allocation_summary.json`.

The current RL reproduction layer is intentionally lightweight: `train_rl.py` implements tabular Q-learning with action masking so training can run on a bare Python install. The paper-style deep-RL path can build from the same environment once `gymnasium`, `torch`, and a trainer such as Stable-Baselines3 or a custom DQN/PPO implementation are added.
