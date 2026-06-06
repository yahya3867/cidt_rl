# CIDT RL and Anomaly Ops Prototype

This repository has two related but separate experiment tracks:

- `src/rl_scheduling/` and `scripts/rl/`: the original workload-placement and reinforcement-learning scheduling prototype.
- `src/anomaly_*`, `src/knowledge_db.py`, `src/slm_client.py`, `src/incident_agent.py`, and root `run_*` anomaly scripts: the Prometheus anomaly detection, SLM extraction, knowledge DB, and incident-ops prototype.

Raw telemetry, generated results, model artifacts, and local SQLite DBs are intentionally excluded from Git.

## RL Scheduling Prototype

The RL scheduling track adapts job-shop and production scheduling ideas to Computing Infrastructure Digital Twin resource allocation. It uses a Prometheus-style server snapshot, generates synthetic jobs, runs baseline placement heuristics, and exports allocation plans that can later be shown in CIDT/Omniverse or used as baselines for a Gymnasium reinforcement-learning environment.

## Run

```powershell
python scripts/rl/run_baselines.py
```

Outputs:

- `data/jobs.csv`
- `results/allocation_plan.csv` for the best-scoring baseline in the current run
- `results/allocation_plan_<heuristic>.csv`
- `results/allocation_evaluation.csv`
- `results/allocation_summary.json`

## Data Mapping

Manufacturing/job-shop scheduling maps into CIDT as:

- Job/task: workload, VM, container, batch job, ML job, or inference job
- Machine: server or node
- Tool/resource constraints: CPU, RAM, disk, network, rack, availability
- Action: assign a workload to a server, delay it, or later migrate it
- Reward/objective: avoid overload, maximize useful utilization, reduce wait time, avoid down machines, balance load, meet deadlines/SLA
- State: server metrics plus queued jobs

## Implemented Baselines

- `random_valid_server`
- `least_loaded_server`
- `most_available_memory`
- `best_fit_resource_match`
- `rack_aware_placement`
- `failure_aware_placement`

All baselines avoid `up=0` machines and reject placements that exceed CPU, memory, or disk capacity.

## Train RL Prototype

```powershell
python scripts/rl/train_q_learning.py --episodes 800
```

The RL layer is a dependency-free tabular Q-learning allocator with action masking. It uses the same state/action/reward idea planned for Gymnasium:

- State: current job demand plus coarse server-capacity buckets
- Action: choose a server index for the current job
- Reward: positive valid placement, penalties for down servers and resource violations, bonus for useful utilization and rack preference

Outputs:

- `results/allocation_plan_q_learning_rl.csv`
- `results/rl_training_summary.json`
- `results/q_table.json`

## Train Stable-Baselines DQN

After creating the local venv and installing requirements:

```powershell
.\.venv\Scripts\python.exe scripts/rl/train_sb3_dqn.py --timesteps 20000
```

The DQN environment chooses from a ranked feasible candidate set instead of all raw servers. This keeps the action space small enough to train on CPU while still producing concrete server placements.

Outputs:

- `results/allocation_plan_dqn_rl.csv`
- `results/dqn_training_summary.json`
- `results/dqn_cidt_model.zip`

## Multi-Snapshot Experiments

```powershell
python scripts/rl/run_snapshot_experiments.py --snapshot-glob "snapshot_*.csv" --job-seeds "42,43,44,45,46"
```

With DQN included:

```powershell
.\.venv\Scripts\python.exe scripts/rl/run_snapshot_experiments.py --snapshot-glob "snapshot_*.csv" --job-seeds "42,43,44" --include-dqn --dqn-timesteps 5000
```

Outputs:

- `results/experiments/experiment_trials.csv`
- `results/experiments/experiment_summary.csv`
- `results/experiments/experiment_summary.json`

## Full Time-Series Experiments

For Prometheus exports where each target has slightly different scrape timestamps, first bucket the history into RL snapshots. Alert windows can optionally be applied as scheduling unavailability:

```powershell
.\.venv\Scripts\python.exe scripts/rl/prepare_prometheus_timeseries.py --metrics C:\path\to\cidt_metrics_history.csv --alert-windows C:\path\to\cidt_alert_windows.csv --output results/rl_cidt_metrics_history/prepared_timeseries.csv
```

```powershell
python scripts/rl/run_timeseries_experiments.py --timeseries data/server_metrics_timeseries.csv --job-seeds "42,43,44"
```

Stress scenarios:

```powershell
python scripts/rl/run_timeseries_experiments.py --scenario mixed_stress --job-count 200 --max-snapshots 100 --job-seeds "42,43,44"
```

Train one DQN policy across sampled Prometheus snapshots:

```powershell
.\.venv\Scripts\python.exe scripts/rl/train_timeseries_dqn.py --timeseries results/rl_cidt_metrics_history/prepared_timeseries.csv --train-snapshots 512 --eval-snapshots 100 --timesteps 250000 --device auto
```

This is the preferred overnight RL path. It samples different telemetry snapshots during training instead of retraining a fresh policy for each timestamp. `--device auto` uses CUDA when the installed PyTorch build can see a GPU.

Available scenarios:

- `normal`
- `cpu_pressure`
- `memory_pressure`
- `disk_pressure`
- `rack_pressure`
- `mixed_stress`

Outputs:

- `results/timeseries_experiments/timeseries_trials.csv`
- `results/timeseries_experiments/timeseries_summary.csv`
- `results/timeseries_experiments/timeseries_snapshot_metadata.csv`
- `results/timeseries_experiments/timeseries_summary.json`

## Prometheus Anomaly Detection Pilot

```powershell
.\.venv\Scripts\python.exe run_anomaly_pilot.py
```

This offline pilot follows the anomaly-detection benchmark structure from `2602.13288v1.pdf` on a small active-host subset of `data/server_metrics_timeseries.csv`. It compares GRU, TCN, Transformer, TSMixer, and Isolation Forest using chronological train/validation/test splits, reconstruction or anomaly scores, rolling likelihood calibration, and weak metric-threshold anomaly windows.

Outputs:

- `results/anomaly_pilot/model_scores.csv`
- `results/anomaly_pilot/detections.csv`
- `results/anomaly_pilot/anomaly_windows.csv`
- `results/anomaly_pilot/summary.json`
- `results/anomaly_pilot/report.md`
- `results/anomaly_pilot/models/*.pt`

## Alert-Window Anomaly Detection Pilot

```powershell
.\.venv\Scripts\python.exe run_alert_anomaly_pilot.py
```

This pilot uses `cidt_metrics_history.csv` as richer telemetry and `cidt_alert_windows.csv` as real critical-alert labels. Models train only on non-alert host telemetry, then evaluate whether GRU, TCN, Transformer, TSMixer, and Isolation Forest separate alerting hosts from healthy hosts in the held-out period.

Run a focused alert-family experiment with:

```powershell
.\.venv\Scripts\python.exe run_alert_anomaly_pilot.py --alert-name HostOutOfDiskSpace --results-dir results/alert_anomaly_disk
```

Outputs:

- `results/alert_anomaly_pilot/model_scores.csv`
- `results/alert_anomaly_pilot/detections.csv`
- `results/alert_anomaly_pilot/host_scores.csv`
- `results/alert_anomaly_pilot/alert_windows.csv`
- `results/alert_anomaly_pilot/summary.json`
- `results/alert_anomaly_pilot/report.md`

## Build Knowledge Database

```powershell
.\.venv\Scripts\python.exe run_build_knowledge_base.py
```

This creates a local SQLite knowledge database from alert-window anomaly results and generates structured metric observations. These observations are the first stand-in for the SLM summary path: each row summarizes alert context, anomaly score evidence, likely telemetry drivers, and a recommended engineering check.

Outputs:

- `results/knowledge/cidt_anomaly_knowledge.db`
- `results/knowledge/observations.csv`

To use LM Studio SLM summaries, start the local server and load three workers with stable identifiers:

```powershell
lms server start
lms load <compatible-model-a> --identifier cidt-slm-a --gpu max --context-length 4096 -y
lms load <compatible-model-b> --identifier cidt-slm-b --gpu max --context-length 4096 -y
lms load <compatible-model-c> --identifier cidt-slm-c --gpu max --context-length 4096 -y
.\.venv\Scripts\python.exe run_build_knowledge_base.py --use-slm
```

The current code round-robins requests across `cidt-slm-a`, `cidt-slm-b`, and `cidt-slm-c`. If LM Studio has no compatible model loaded, the pipeline keeps the deterministic summary and records the SLM failure in `evidence_json`.

## Ops Dashboard and API

```powershell
.\.venv\Scripts\python.exe run_ops_server.py --agent-model google/gemma-3-4b
```

Open `http://127.0.0.1:8765` for a simple local dashboard. The ops layer is log-entry based: hosts are metadata, while alerts, anomaly scores, SLM observations, feedback, and incident reports are tied to individual `log_entries`.

- `GET /api/log-entries`
- `GET /api/observations`
- `GET /api/anomalies`
- `POST /api/incidents/generate` with `{"log_entry_id": 25}` or `{"instance": "..."}`
- `POST /api/feedback` with `{"log_entry_id": 25, "feedback_type": "true_positive|false_positive|known_maintenance|duplicate|resolved", "note": "..."}`
