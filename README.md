# Capstone Economy RL

This project trains two PyTorch actor-critic policies in the archived AI Economist Foundation environment:

- A shared worker policy chooses a fresh action for every worker on every timestep.
- A planner policy chooses a new economic policy once every 30 worker timesteps.
- The planner action is held fixed between decision boundaries.
- Four independent 25x25 worlds are interleaved in one training process.
- Each world contains 50 worker agents.
- Worker rewards are stored per timestep; planner rewards are aggregated over each 30-step interval.
- Episodes reset independently when their `done` flag is returned.
- Checkpoints and JSONL training history are written under `outputs/`.

## How the economy is simulated

Each world is a 25x25 grid containing 50 mobile worker agents and one social planner. The `Gather` component lets workers collect wood and stone, while `Build` lets them spend resources on construction. At each worker timestep, the worker policy observes its local state and chooses one discrete action. The planner policy observes the planner state and chooses a policy action only at timestep 0 and then every 30 worker timesteps; that action is held fixed between planner decisions. The environment applies all actions, updates resources/buildings/agent state, and returns one reward per agent plus a planner reward.

Four worlds are interleaved independently. They share policy weights but have separate maps, episode counters, rewards, and terminal resets. PPO uses worker transitions at every timestep and planner transitions over each 30-step interval. This is a hierarchical or two-timescale simulation, not a single-agent economic model.

## Run with Docker

```bash
docker build -t capstone-economy-cuda .
docker run --rm --gpus all -it -v "$PWD":/workspace capstone-economy-cuda bash
python3.8 -c "import torch; print(torch.cuda.is_available())"
python -m economy_rl.training.smoke
python -m economy_rl.training.train --config configs/default.json
```

## Dump and analyze data

Training writes copyable files under `outputs/data/`:

- `steps.csv`: one row per world and worker timestep, including planner reward, total/mean worker reward, action, and normalized reward gap.
- `planner_intervals.csv`: one row per planner decision interval.
- `episodes.csv`: completed episode totals by world.
- `updates.csv`: PPO losses and entropy by update.
- `summary.json`: configuration and run counters.

Generate analysis files after training:

```bash
python3.8 -m economy_rl.analysis.analyze \
	--data-dir outputs/data \
	--output-dir outputs/analysis
```

The trainer also runs this analysis automatically after the final update. During training it prints one `planner_cycle` line whenever a 30-worker-step planner interval closes. Each line includes the world, episode, selected policy, interval length, and interval reward.

The post-training console output includes:

```text
training_complete
graph=outputs/analysis/economy_training.png
fit_status=learning_signal|possible_underfit|uncertain
```

The fit check compares early and late worker rewards, PPO loss, and policy entropy. It cannot prove overfitting from training data alone; the report explicitly says `overfit_status: not_testable_without_held_out_evaluation` until a separate fresh-episode evaluation is supplied.

The analysis directory contains CSV files with rolling metrics, `economy_training.png`, and `convergence.json`. Copy `outputs/data/` or `outputs/analysis/` out of the container with:

```bash
docker cp capstone-economy-cuda:/workspace/outputs ./outputs-from-container
```

The convergence report measures reward alignment, not proof of an economic equilibrium. It compares the planner reward with the mean worker reward using:

```text
abs(planner_reward - worker_mean_reward)
--------------------------------------
abs(planner_reward) + abs(worker_mean_reward) + 1e-8
```

It reports convergence only when the final rolling gap is at least 25% lower than the initial gap and its recent variation is small. A long training run is required before interpreting this result.

The package is under `src/`, so set the import path inside the container:

```bash
export PYTHONPATH=/workspace/src
```

From the repository on the host, use the same path when running a local command:

```bash
PYTHONPATH=src python3 test.py
```

For a quick run, lower `updates` and `rollout_steps` in `configs/default.json`. The requested configuration represents 200 workers across four simulations and can be computationally expensive.

## Outputs

- `outputs/history.jsonl`: planner interval and PPO update metrics.
- `outputs/checkpoints/update_XXXXXX.pt`: planner and worker weights, optimizers, configuration, and counters.

Evaluate a checkpoint with:

```bash
python -m economy_rl.training.evaluate outputs/checkpoints/update_000000.pt
```

## Compatibility note

AI Economist was archived and targets an older Python ecosystem. The smoke command should be run before training. It reports the actual planner action dimensions, worker action space, reward keys, and observation-agent count discovered from the installed package.
