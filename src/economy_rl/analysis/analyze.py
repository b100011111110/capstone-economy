import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def _read_table(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def convergence_report(steps: pd.DataFrame) -> Dict[str, object]:
    if steps.empty:
        return {"status": "insufficient_data", "reason": "steps.csv is empty or missing"}
    gap = steps["reward_gap"].astype(float)
    window = max(1, min(50, len(gap) // 5))
    initial = float(gap.iloc[:window].mean())
    final = float(gap.iloc[-window:].mean())
    final_std = float(gap.iloc[-window:].std(ddof=0))
    reduction = (initial - final) / max(initial, 1e-8)
    converged = bool(reduction >= 0.25 and final_std <= 0.10)
    return {
        "status": "converged" if converged else "not_converged",
        "metric": "normalized_planner_worker_reward_gap",
        "definition": "abs(planner_reward - worker_mean_reward) / (abs(planner_reward) + abs(worker_mean_reward) + 1e-8)",
        "initial_mean_gap": initial,
        "final_mean_gap": final,
        "final_gap_std": final_std,
        "gap_reduction": reduction,
        "window_steps": window,
        "interpretation": (
            "The planner and worker rewards became closer and stable under this metric."
            if converged else
            "The available run does not show a sufficiently large and stable reward-gap reduction."
        ),
        "warning": "This is reward alignment, not proof of a Nash equilibrium or economic optimality.",
    }


def fit_report(steps: pd.DataFrame, updates: pd.DataFrame) -> Dict[str, object]:
    """Give a conservative fit check; overfitting needs a held-out evaluation set."""
    if steps.empty or updates.empty:
        return {
            "fit_status": "insufficient_data",
            "fit_explanation": "There is not enough step and update data to assess learning.",
        }
    window = max(1, min(50, len(steps) // 5))
    initial_reward = float(steps.worker_mean_reward.iloc[:window].mean())
    final_reward = float(steps.worker_mean_reward.iloc[-window:].mean())
    initial_loss = float(updates.worker_loss.iloc[0])
    final_loss = float(updates.worker_loss.iloc[-1])
    initial_entropy = float(updates.worker_entropy.iloc[0])
    final_entropy = float(updates.worker_entropy.iloc[-1])
    improving = final_reward > initial_reward and final_loss < initial_loss
    underfit = final_reward <= initial_reward and final_entropy > 1.0
    return {
        "fit_status": "possible_underfit" if underfit else ("learning_signal" if improving else "uncertain"),
        "fit_explanation": (
            "Worker reward and PPO loss improved, so the run contains a learning signal; this is not an overfitting test."
            if improving else
            "The run does not show a clear reward/loss improvement; more training or a better reward signal may be needed."
            if underfit else
            "Training-only metrics are inconclusive. Overfitting requires evaluation on fresh held-out episodes."
        ),
        "initial_worker_reward": initial_reward,
        "final_worker_reward": final_reward,
        "initial_worker_loss": initial_loss,
        "final_worker_loss": final_loss,
        "initial_worker_entropy": initial_entropy,
        "final_worker_entropy": final_entropy,
        "overfit_status": "not_testable_without_held_out_evaluation",
    }


def analyze(data_dir: str, output_dir: str) -> Tuple[Path, Dict[str, object]]:
    source = Path(data_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    steps = _read_table(source, "steps.csv")
    episodes = _read_table(source, "episodes.csv")
    updates = _read_table(source, "updates.csv")
    if steps.empty:
        report = convergence_report(steps)
        with (destination / "convergence.json").open("w") as handle:
            json.dump(report, handle, indent=2)
        return destination, report

    steps = steps.sort_values("global_step")
    steps["rolling_planner_reward"] = steps["planner_reward"].rolling(20, min_periods=1).mean()
    steps["rolling_worker_mean_reward"] = steps["worker_mean_reward"].rolling(20, min_periods=1).mean()
    steps["rolling_reward_gap"] = steps["reward_gap"].rolling(20, min_periods=1).mean()
    steps["rolling_worker_action_mean"] = steps["worker_action_mean"].rolling(20, min_periods=1).mean()
    steps.to_csv(destination / "steps_with_rolling_metrics.csv", index=False)
    if not episodes.empty:
        episodes.to_csv(destination / "episodes.csv", index=False)
    if not updates.empty:
        updates.to_csv(destination / "updates.csv", index=False)

    report = convergence_report(steps)
    report.update(fit_report(steps, updates))
    report["steps"] = int(len(steps))
    report["episodes"] = int(len(episodes))
    report["worlds"] = sorted(int(value) for value in steps["world"].unique())
    with (destination / "convergence.json").open("w") as handle:
        json.dump(report, handle, indent=2)

    figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(steps["global_step"], steps["rolling_planner_reward"], label="Planner reward", color="#b23a48")
    axes[0].plot(steps["global_step"], steps["rolling_worker_mean_reward"], label="Mean worker reward", color="#2364aa")
    axes[0].set_ylabel("Rolling reward")
    axes[0].set_title("Planner and Worker Rewards")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps["global_step"], steps["rolling_reward_gap"], label="Normalized reward gap", color="#2a9d8f")
    axes[1].axhline(0.10, color="#777777", linestyle="--", linewidth=1, label="0.10 stability reference")
    axes[1].set_xlabel("Global worker step")
    axes[1].set_ylabel("Gap")
    axes[1].set_title("Planner-Worker Reward Alignment")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    axes[2].step(steps["global_step"], steps["planner_action_value"], where="post", label="Planner action", color="#e76f51")
    axes[2].plot(steps["global_step"], steps["rolling_worker_action_mean"], label="Rolling mean worker action", color="#264653")
    axes[2].set_xlabel("Global worker step")
    axes[2].set_ylabel("Action value")
    axes[2].set_title("Worker Action Response to Planner Policy")
    axes[2].legend()
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination / "economy_training.png", dpi=160)
    plt.close(figure)
    return destination, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dumped economy training data")
    parser.add_argument("--data-dir", default="outputs/data")
    parser.add_argument("--output-dir", default="outputs/analysis")
    args = parser.parse_args()
    destination, report = analyze(args.data_dir, args.output_dir)
    print(json.dumps({"analysis_directory": str(destination), **report}, indent=2))


if __name__ == "__main__":
    main()
