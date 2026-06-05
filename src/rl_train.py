from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict

from .evaluate import summarize
from .models import Allocation, Job, Server
from .rl_env import CidtAllocationEnv

State = tuple[int, ...]
QTable = dict[State, list[float]]


def train_q_learning(
    servers: list[Server],
    jobs: list[Job],
    episodes: int = 800,
    seed: int = 42,
    learning_rate: float = 0.18,
    discount: float = 0.92,
    epsilon_start: float = 0.55,
    epsilon_end: float = 0.05,
) -> tuple[QTable, list[dict[str, float]]]:
    rng = random.Random(seed)
    env = CidtAllocationEnv(servers, jobs)
    q_table: QTable = defaultdict(lambda: [0.0] * env.action_count)
    metrics: list[dict[str, float]] = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        assigned = 0
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(0.0, 1.0 - episode / max(episodes, 1))

        while not done:
            action = _choose_action(q_table, state, env.valid_actions(), env.action_count, epsilon, rng)
            next_state, reward, done, info = env.step(action)
            allocation: Allocation = info["allocation"]  # type: ignore[assignment]
            assigned += 1 if allocation.assigned else 0
            total_reward += reward

            current = q_table[state][action]
            next_best = max(q_table[next_state]) if not done else 0.0
            q_table[state][action] = current + learning_rate * (reward + discount * next_best - current)
            state = next_state

        if episode == 1 or episode % 25 == 0 or episode == episodes:
            metrics.append(
                {
                    "episode": float(episode),
                    "epsilon": round(epsilon, 4),
                    "total_reward": round(total_reward, 4),
                    "assignment_rate": round(assigned / max(len(jobs), 1), 4),
                }
            )

    return dict(q_table), metrics


def evaluate_q_policy(
    servers: list[Server],
    jobs: list[Job],
    q_table: QTable,
) -> tuple[list[Allocation], dict[str, object]]:
    env = CidtAllocationEnv(servers, jobs)
    state, _ = env.reset()
    done = False
    allocations: list[Allocation] = []

    while not done:
        action = _choose_action(q_table, state, env.valid_actions(), env.action_count, 0.0, random.Random(0))
        state, _, done, info = env.step(action)
        allocations.append(info["allocation"])  # type: ignore[arg-type]

    return allocations, summarize(allocations, jobs)


def serializable_q_table(q_table: QTable) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state, values in sorted(q_table.items(), key=lambda item: item[0]):
        best_action = max(range(len(values)), key=lambda index: values[index]) if values else -1
        rows.append(
            {
                "state": list(state),
                "best_action": best_action,
                "best_value": round(values[best_action], 6) if best_action >= 0 else 0.0,
                "values": [round(value, 6) for value in values],
            }
        )
    return rows


def allocations_as_dicts(allocations: list[Allocation]) -> list[dict[str, object]]:
    return [asdict(allocation) for allocation in allocations]


def _choose_action(
    q_table: QTable,
    state: State,
    valid_actions: list[int],
    action_count: int,
    epsilon: float,
    rng: random.Random,
) -> int:
    if rng.random() < epsilon:
        if valid_actions and rng.random() < 0.9:
            return rng.choice(valid_actions)
        return rng.randrange(action_count)

    candidates = valid_actions or list(range(action_count))
    values = q_table[state]
    return max(candidates, key=lambda action: values[action])
