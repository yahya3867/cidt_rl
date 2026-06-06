# Full-Dataset RL Scheduling Results

## Results

We evaluated eight scheduling methods on the full prepared Prometheus telemetry dataset derived from `cidt_metrics_history.csv` and `cidt_alert_windows.csv`. The metrics history was bucketed into 5-minute snapshots, and alert windows were applied as host unavailability signals. The final prepared dataset contained 2,899 time snapshots, 327,587 usable host samples, 113 hosts with complete resource telemetry, and 49 applied alert windows.

Each method was evaluated over 8,697 trials, corresponding to 2,899 snapshots and three workload seeds. The comparison included six heuristic schedulers, one tabular Q-learning method, and one GPU-trained deep reinforcement learning method (`timeseries_dqn_rl`). The DQN model was trained for 1,000,000 timesteps across all 2,899 snapshots using CUDA on an NVIDIA GeForce RTX 4060, then evaluated across all snapshots and workload seeds.

| Rank | Method | Type | Trials | Assignment Rate | Mean Score | Failed Jobs Mean | CPU Stddev Mean | Servers Used Mean | Racks Used Mean |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `failure_aware_placement` | Heuristic | 8,697 | 1.000 | 19.0646 | 0.000 | 20.2301 | 14.6680 | 3.3347 |
| 2 | `most_available_memory` | Heuristic | 8,697 | 1.000 | 18.8800 | 0.000 | 21.4297 | 23.0000 | 3.0000 |
| 3 | `timeseries_dqn_rl` | Deep RL, DQN | 8,697 | 1.000 | 18.8151 | 0.000 | 16.1277 | 17.6751 | 3.7892 |
| 4 | `rack_aware_placement` | Heuristic | 8,697 | 1.000 | 18.7845 | 0.000 | 19.8444 | 46.8547 | 4.9993 |
| 5 | `least_loaded_server` | Heuristic | 8,697 | 1.000 | 18.7702 | 0.000 | 19.4956 | 47.9998 | 4.9995 |
| 6 | `random_valid_server` | Heuristic | 8,697 | 1.000 | 18.7700 | 0.000 | 19.7475 | 37.0000 | 4.6667 |
| 7 | `best_fit_resource_match` | Heuristic | 8,697 | 1.000 | 18.6957 | 0.000 | 23.5543 | 19.2013 | 3.3293 |
| 8 | `q_learning_rl` | Tabular RL | 8,697 | 1.000 | 16.5010 | 0.000 | 24.1277 | 17.8726 | 4.5588 |

All methods achieved a 100% assignment rate and zero failed jobs under the evaluated workload. This indicates that the generated workload demand was feasible across the available host pool, even after alert-window hosts were marked unavailable. Because assignment rate was saturated, the differentiating metric was the mean placement score, which captures resource fit, priority handling, utilization quality, and scheduling risk penalties.

The strongest method was `failure_aware_placement`, with a mean score of 19.0646. This suggests that a hand-designed heuristic that explicitly avoids unavailable or risky hosts remains highly competitive for this workload. `most_available_memory` ranked second, indicating that memory headroom is an important scheduling signal in this telemetry slice.

The DQN-based deep RL method ranked third overall, with a mean score of 18.8151. Although it did not exceed the best heuristic, it outperformed four of the six heuristic baselines and substantially outperformed tabular Q-learning. It also achieved the lowest CPU-after-placement standard deviation among the top methods, suggesting that the learned policy produced comparatively smoother CPU utilization. This is a useful result: the neural policy learned a viable placement strategy from the time-series environment, but additional reward shaping, longer training, and broader workload scenarios may be needed before it surpasses the strongest engineered heuristic.

Tabular Q-learning ranked last, with a mean score of 16.5010. This result is expected because the tabular state representation is coarse and does not generalize well across the full telemetry history. In contrast, the DQN policy receives continuous state features and can generalize across snapshots, which explains its substantially stronger performance.

## Interpretation

The results show that the full-dataset RL environment is operational and that GPU-trained DQN can learn a competitive scheduling policy from Prometheus-derived telemetry. However, the current best-performing scheduler remains the failure-aware heuristic. The primary research implication is that deep RL is promising but not yet dominant in this setup. The next experimental step should focus on making the DRL task harder and more realistic: mixed workload scenarios, stronger alert penalties, time-varying job arrivals, and reward terms that explicitly optimize resilience under incident conditions.

## Artifacts

- Prepared full telemetry input: `results/rl_cidt_metrics_history/prepared_timeseries.csv`
- Heuristic and tabular RL results: `results/rl_cidt_metrics_history/all_methods_full_q/`
- Full DQN results: `results/rl_cidt_metrics_history/timeseries_dqn_full_1m/`
- Trained DQN model: `results/rl_cidt_metrics_history/timeseries_dqn_full_1m/timeseries_dqn_model.zip`
