from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.rl_scheduling.evaluate import summarize
from src.rl_scheduling.export_results import export_allocation_plan, export_summary
from src.rl_scheduling.gym_env import CidtGymEnv
from src.rl_scheduling.loaders import load_jobs, load_servers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Stable-Baselines3 DQN allocator for CIDT.")
    parser.add_argument("--servers", default="data/servers_snapshot.csv")
    parser.add_argument("--jobs", default="data/jobs.csv")
    parser.add_argument("--timesteps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-path", default="results/dqn_cidt_model.zip")
    parser.add_argument("--check-env", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    servers = load_servers(args.servers)
    jobs = load_jobs(args.jobs)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    env = CidtGymEnv(servers, jobs)
    if args.check_env:
        check_env(env, warn=True)

    train_env = Monitor(CidtGymEnv(servers, jobs), filename=str(results_dir / "dqn_monitor.csv"))
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=2.5e-4,
        buffer_size=50000,
        learning_starts=500,
        batch_size=64,
        gamma=0.96,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.35,
        exploration_initial_eps=0.90,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        verbose=0,
        seed=args.seed,
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    model.save(args.model_path)

    allocations = evaluate_policy(model, CidtGymEnv(servers, jobs))
    summary = summarize(allocations, jobs)
    export_allocation_plan(results_dir / "allocation_plan_dqn_rl.csv", allocations)
    export_summary(
        results_dir / "dqn_training_summary.json",
        {
            "algorithm": "stable_baselines3_dqn",
            "timesteps": args.timesteps,
            "seed": args.seed,
            "server_snapshot": args.servers,
            "jobs": args.jobs,
            "model_path": args.model_path,
            "evaluation_summary": summary,
            "note": "This DQN chooses from a ranked feasible candidate set instead of all raw servers, which keeps the action space trainable.",
        },
    )

    print(f"Trained DQN for {args.timesteps} timesteps.")
    print(
        "Evaluation: "
        f"assignment_rate={summary['assignment_rate']} "
        f"average_score={summary['average_score']} "
        f"jobs_failed={summary['jobs_failed']}"
    )
    print(f"Saved model to {Path(args.model_path).resolve()}.")


def evaluate_policy(model: DQN, env: CidtGymEnv):
    observations, _ = env.reset()
    done = False
    allocations = []
    while not done:
        action, _ = model.predict(observations, deterministic=True)
        observations, _, terminated, truncated, info = env.step(int(action))
        allocation = info.get("allocation")
        if allocation is not None:
            allocations.append(allocation)
        done = terminated or truncated
    return allocations


if __name__ == "__main__":
    main()
