"""
Comprehensive training visualization script for Economy RL.
Loads step-level, episode-level, and update-level metrics and produces high-resolution plots.
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(data_dir: Path):
    steps_path = data_dir / "steps.csv"
    episodes_path = data_dir / "episodes.csv"
    updates_path = data_dir / "updates.csv"
    convergence_path = data_dir / "convergence.json"
    
    # Fallback to analysis dir if needed
    if not steps_path.exists() and (data_dir / "steps_with_rolling_metrics.csv").exists():
        steps_path = data_dir / "steps_with_rolling_metrics.csv"
        
    steps = pd.read_csv(steps_path) if steps_path.exists() else pd.DataFrame()
    episodes = pd.read_csv(episodes_path) if episodes_path.exists() else pd.DataFrame()
    updates = pd.read_csv(updates_path) if updates_path.exists() else pd.DataFrame()
    
    convergence = {}
    if convergence_path.exists():
        with open(convergence_path, "r") as f:
            convergence = json.load(f)
            
    return steps, episodes, updates, convergence


def generate_training_plots(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    steps, episodes, updates, convergence = load_data(data_dir)
    
    print(f"Loaded: steps={len(steps)}, episodes={len(episodes)}, updates={len(updates)}")
    
    # Set overall styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 14,
        "figure.autolayout": False,
        "lines.linewidth": 1.5,
    })

    # Subsample steps if very large for smooth plotting performance (400k points)
    if len(steps) > 50000:
        step_stride = max(1, len(steps) // 20000)
        plot_steps = steps.iloc[::step_stride].copy()
    else:
        plot_steps = steps.copy()

    # Calculate rolling metrics if not present
    if "rolling_planner_reward" not in plot_steps.columns and "planner_reward" in plot_steps.columns:
        plot_steps["rolling_planner_reward"] = plot_steps["planner_reward"].rolling(50, min_periods=1).mean()
    if "rolling_worker_mean_reward" not in plot_steps.columns and "worker_mean_reward" in plot_steps.columns:
        plot_steps["rolling_worker_mean_reward"] = plot_steps["worker_mean_reward"].rolling(50, min_periods=1).mean()
    if "rolling_reward_gap" not in plot_steps.columns and "reward_gap" in plot_steps.columns:
        plot_steps["rolling_reward_gap"] = plot_steps["reward_gap"].rolling(50, min_periods=1).mean()
    if "rolling_worker_action_mean" not in plot_steps.columns and "worker_action_mean" in plot_steps.columns:
        plot_steps["rolling_worker_action_mean"] = plot_steps["worker_action_mean"].rolling(50, min_periods=1).mean()

    # ==========================================
    # 1. Master Training Dashboard (6-panel)
    # ==========================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 14), sharex=False)
    fig.suptitle("Economy RL Multi-Agent Training Dashboard (100k Iterations / 400k Steps)", fontsize=16, fontweight="bold", y=0.98)

    # 1.1 Rewards over global steps
    ax = axes[0, 0]
    if not plot_steps.empty and "planner_reward" in plot_steps.columns:
        ax.plot(plot_steps["global_step"], plot_steps["rolling_planner_reward"], label="Planner Reward (Rolling Mean)", color="#d62728", linewidth=2)
        ax.plot(plot_steps["global_step"], plot_steps["rolling_worker_mean_reward"], label="Worker Mean Reward (Rolling Mean)", color="#1f77b4", linewidth=2)
        ax.set_title("1. Planner & Worker Rewards over Steps", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Step Reward")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 1.2 Episode Total Rewards
    ax = axes[0, 1]
    if not episodes.empty:
        # Sort and rolling average across episodes
        episodes_sorted = episodes.sort_values("global_step").reset_index(drop=True)
        episodes_sorted["ep_idx"] = np.arange(len(episodes_sorted))
        episodes_sorted["roll_worker_tot"] = episodes_sorted["worker_total_reward"].rolling(30, min_periods=1).mean()
        episodes_sorted["roll_planner_tot"] = episodes_sorted["planner_total_reward"].rolling(30, min_periods=1).mean()
        
        ax.plot(episodes_sorted["global_step"], episodes_sorted["roll_planner_tot"], label="Planner Total Return (Rolling)", color="#e377c2", linewidth=2)
        ax.plot(episodes_sorted["global_step"], episodes_sorted["roll_worker_tot"], label="Worker Total Return (Rolling)", color="#2ca02c", linewidth=2)
        ax.set_title("2. Episode Cumulative Returns", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Episode Return")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 1.3 Reward Alignment (Gap)
    ax = axes[1, 0]
    if not plot_steps.empty and "reward_gap" in plot_steps.columns:
        ax.plot(plot_steps["global_step"], plot_steps["rolling_reward_gap"], label="Normalized Reward Gap", color="#ff7f0e", linewidth=2)
        ax.axhline(0.10, color="green", linestyle="--", alpha=0.7, label="Convergence Threshold (0.10)")
        ax.set_title("3. Planner-Worker Reward Alignment (Gap)", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Normalized Gap")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 1.4 Worker Policy & Value Losses
    ax = axes[1, 1]
    if not updates.empty:
        ax.plot(updates["global_step"], updates["worker_loss"], label="Total Worker Loss", color="#17becf", linewidth=2)
        ax.plot(updates["global_step"], updates["worker_value_loss"], label="Worker Value Loss", color="#bcbd22", linestyle="--", linewidth=1.5)
        ax.plot(updates["global_step"], updates["worker_policy_loss"], label="Worker Policy Loss", color="#9467bd", linestyle=":", linewidth=1.5)
        ax.set_title("4. Worker PPO Loss Breakdown", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Loss")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 1.5 Policy Entropy (Exploration Decay)
    ax = axes[2, 0]
    if not updates.empty:
        ax.plot(updates["global_step"], updates["worker_entropy"], label="Worker Policy Entropy", color="#1f77b4", linewidth=2)
        ax.plot(updates["global_step"], updates["planner_entropy"], label="Planner Policy Entropy", color="#d62728", linewidth=2)
        ax.set_title("5. Policy Entropy (Exploration Decay)", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Entropy (nats)")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    # 1.6 Policy Actions Dynamics
    ax = axes[2, 1]
    if not plot_steps.empty and "planner_action_value" in plot_steps.columns:
        ax.step(plot_steps["global_step"], plot_steps["planner_action_value"], where="post", label="Planner Policy Action (Tax Rate / Bracket)", color="#e76f51", alpha=0.7, linewidth=1.2)
        ax.plot(plot_steps["global_step"], plot_steps["rolling_worker_action_mean"], label="Mean Worker Labor Action (Rolling)", color="#264653", linewidth=2)
        ax.set_title("6. Planner Policy Actions & Worker Response", fontweight="bold")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Action Value")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    dashboard_path = output_dir / "training_dashboard.png"
    fig.savefig(dashboard_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {dashboard_path}")

    # ==========================================
    # 2. Detailed 3-Panel Vertical View (Economy Training Overview)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Economy RL: Rewards, Alignment & Policy Evolution", fontsize=15, fontweight="bold")
    
    # Rewards
    axes[0].plot(plot_steps["global_step"], plot_steps["rolling_planner_reward"], label="Planner Reward (Rolling)", color="#b23a48", linewidth=2)
    axes[0].plot(plot_steps["global_step"], plot_steps["rolling_worker_mean_reward"], label="Mean Worker Reward (Rolling)", color="#2364aa", linewidth=2)
    axes[0].set_ylabel("Rolling Reward", fontweight="bold")
    axes[0].set_title("Planner and Worker Step Rewards")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Alignment
    axes[1].plot(plot_steps["global_step"], plot_steps["rolling_reward_gap"], label="Normalized Reward Gap", color="#2a9d8f", linewidth=2)
    axes[1].axhline(0.10, color="#777777", linestyle="--", linewidth=1.2, label="Convergence Threshold (0.10)")
    axes[1].set_ylabel("Reward Gap", fontweight="bold")
    axes[1].set_title("Planner-Worker Reward Alignment Metric")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    # Actions
    axes[2].step(plot_steps["global_step"], plot_steps["planner_action_value"], where="post", label="Planner Action", color="#e76f51", alpha=0.8, linewidth=1.2)
    axes[2].plot(plot_steps["global_step"], plot_steps["rolling_worker_action_mean"], label="Rolling Mean Worker Action", color="#264653", linewidth=2)
    axes[2].set_xlabel("Global Worker Step", fontweight="bold")
    axes[2].set_ylabel("Action Value", fontweight="bold")
    axes[2].set_title("Worker Action Response to Planner Policy")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    detailed_path = output_dir / "economy_training_detailed.png"
    fig.savefig(detailed_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {detailed_path}")

    # ==========================================
    # 3. PPO Optimization Metrics (Losses & Entropies)
    # ==========================================
    if not updates.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        fig.suptitle("PPO Algorithm Optimization & Convergence Dynamics", fontsize=15, fontweight="bold")

        # Worker Loss
        axes[0, 0].plot(updates["global_step"], updates["worker_loss"], color="#1f77b4", label="Total Worker Loss", linewidth=2)
        axes[0, 0].plot(updates["global_step"], updates["worker_value_loss"], color="#aec7e8", label="Value Loss", linestyle="--")
        axes[0, 0].set_title("Worker PPO Total & Value Loss")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Worker Policy Loss
        axes[0, 1].plot(updates["global_step"], updates["worker_policy_loss"], color="#ff7f0e", label="Worker Policy Loss", linewidth=2)
        axes[0, 1].set_title("Worker PPO Policy (Clip) Loss")
        axes[0, 1].set_ylabel("Policy Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Planner Loss
        axes[1, 0].plot(updates["global_step"], updates["planner_loss"], color="#d62728", label="Total Planner Loss", linewidth=2)
        axes[1, 0].set_title("Planner PPO Total Loss")
        axes[1, 0].set_xlabel("Global Step")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Entropies
        axes[1, 1].plot(updates["global_step"], updates["worker_entropy"], color="#2ca02c", label="Worker Entropy", linewidth=2)
        axes[1, 1].plot(updates["global_step"], updates["planner_entropy"], color="#9467bd", label="Planner Entropy", linewidth=2)
        axes[1, 1].set_title("Worker & Planner Policy Entropies")
        axes[1, 1].set_xlabel("Global Step")
        axes[1, 1].set_ylabel("Entropy (nats)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        ppo_path = output_dir / "ppo_optimization_dynamics.png"
        fig.savefig(ppo_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {ppo_path}")

    # ==========================================
    # 4. Democratic Political Economy & Loyalty Dashboard
    # ==========================================
    loyalty_path = data_dir / "loyalty.csv"
    elections_path = data_dir / "elections.csv"
    if loyalty_path.exists() or elections_path.exists():
        loyalty_df = pd.read_csv(loyalty_path) if loyalty_path.exists() else pd.DataFrame()
        elections_df = pd.read_csv(elections_path) if elections_path.exists() else pd.DataFrame()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Democratic Political Economy & Dynamic Voter Loyalty [-1.0, +1.0]", fontsize=16, fontweight="bold")

        # 4.1 Average Voter Loyalty trajectories per Leader
        ax = axes[0, 0]
        if not loyalty_df.empty:
            l_step = loyalty_df.iloc[::max(1, len(loyalty_df) // 5000)].copy()
            l_step["l0_roll"] = l_step["leader_0_loyalty"].rolling(50, min_periods=1).mean()
            l_step["l1_roll"] = l_step["leader_1_loyalty"].rolling(50, min_periods=1).mean()
            l_step["l2_roll"] = l_step["leader_2_loyalty"].rolling(50, min_periods=1).mean()
            ax.plot(l_step["global_step"], l_step["l0_roll"], label="Leader 0 (Candidate A)", color="#1f77b4", linewidth=2)
            ax.plot(l_step["global_step"], l_step["l1_roll"], label="Leader 1 (Candidate B)", color="#ff7f0e", linewidth=2)
            ax.plot(l_step["global_step"], l_step["l2_roll"], label="Leader 2 (Candidate C)", color="#2ca02c", linewidth=2)
            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.7, label="Indifference (0.0)")
            ax.set_title("Voter Loyalty Evolution (Approval [-1, +1])", fontweight="bold")
            ax.set_xlabel("Global Step")
            ax.set_ylabel("Mean Voter Loyalty")
            ax.set_ylim(-1.05, 1.05)
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)

        # 4.2 Election Vote Shares over Time
        ax = axes[0, 1]
        if not elections_df.empty and "vote_shares" in elections_df.columns:
            try:
                shares = [json.loads(s) for s in elections_df["vote_shares"]]
                shares_df = pd.DataFrame(shares, columns=["Candidate 0", "Candidate 1", "Candidate 2"])
                shares_df["global_step"] = elections_df["global_step"]
                shares_df = shares_df.sort_values("global_step").reset_index(drop=True)
                for col in ["Candidate 0", "Candidate 1", "Candidate 2"]:
                    shares_df[f"{col}_roll"] = shares_df[col].rolling(30, min_periods=1).mean()
                
                ax.plot(shares_df["global_step"], shares_df["Candidate 0_roll"], label="Leader 0 (Broad Coalition)", color="#1f77b4", linewidth=2)
                ax.plot(shares_df["global_step"], shares_df["Candidate 1_roll"], label="Leader 1 (Fiscal Conservative)", color="#ff7f0e", linewidth=2)
                ax.plot(shares_df["global_step"], shares_df["Candidate 2_roll"], label="Leader 2 (Progressive Welfare)", color="#2ca02c", linewidth=2)
                ax.axhline(0.333, color="gray", linestyle=":", alpha=0.7, label="Equal Split (33%)")
                ax.set_title("Democratic Election Vote Shares (%)", fontweight="bold")
                ax.set_xlabel("Global Step")
                ax.set_ylabel("Vote Share (Rolling Mean)")
                ax.set_ylim(-0.05, 1.05)
                ax.legend(loc="best")
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Could not parse vote shares: {e}")

        # 4.3 Dynamic Demographic Cluster Shares
        ax = axes[1, 0]
        if not loyalty_df.empty and "cluster_proportions" in loyalty_df.columns:
            try:
                c_props = [json.loads(p) for p in loyalty_df["cluster_proportions"] if p]
                if c_props:
                    c_df = pd.DataFrame(c_props, columns=["Cluster 0", "Cluster 1", "Cluster 2"])
                    c_df["global_step"] = loyalty_df["global_step"].iloc[:len(c_df)]
                    c_sub = c_df.iloc[::max(1, len(c_df) // 1000)].copy()
                    c_grouped = c_sub.groupby("global_step").mean().reset_index()
                    ax.stackplot(
                        c_grouped["global_step"],
                        c_grouped["Cluster 0"],
                        c_grouped["Cluster 1"],
                        c_grouped["Cluster 2"],
                        labels=["Dynamic Cluster 0", "Dynamic Cluster 1", "Dynamic Cluster 2"],
                        colors=["#aec7e8", "#ffbb78", "#98df8a"],
                        alpha=0.8,
                    )
                    ax.set_title("Dynamic N-Dimensional Agent Clustering Evolution", fontweight="bold")
                    ax.set_xlabel("Global Step")
                    ax.set_ylabel("Population Share")
                    ax.set_ylim(0, 1.0)
                    ax.legend(loc="upper right")
                    ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"Could not parse cluster proportions: {e}")

        # 4.4 Election Winners & Turnover
        ax = axes[1, 1]
        if not elections_df.empty and "winner_id" in elections_df.columns:
            winner_counts = elections_df["winner_id"].value_counts()
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
            ax.bar([f"Leader {i}" for i in range(3)], [winner_counts.get(i, 0) for i in range(3)], color=colors, alpha=0.85)
            ax.set_title(f"Total Election Wins (Total Elections: {len(elections_df)})", fontweight="bold")
            ax.set_ylabel("Elections Won")
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        pol_path = output_dir / "political_economy_dashboard.png"
        fig.savefig(pol_path, dpi=200)
        plt.close(fig)
        print(f"Saved: {pol_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive training plots")
    parser.add_argument("--data-dir", default="outputs/100k/data", help="Path to input data directory")
    parser.add_argument("--output-dir", default="outputs/training_plots", help="Path to output plot directory")
    args = parser.parse_args()
    
    generate_training_plots(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

