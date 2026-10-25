#!/usr/bin/env python3
"""Generate publication-grade comparative graphs between Experiment 7 (Voting) and Experiment 8 (Central Planner).

Compares:
1. Wealth Generated over each episode
2. Gini (Wealth Inequality) over each episode
3. Houses Built over each episode
4. Resources Accumulated / Mined over each episode

Outputs both a combined 4-panel dashboard and individual standalone graphs for each metric.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Styling palette
COLOR_EXP7 = "#1f77b4"      # Democratic Voting (Blue)
COLOR_EXP7_FILL = "#a6c8e0"
COLOR_EXP8 = "#d62728"      # Central Planning (Red)
COLOR_EXP8_FILL = "#f5b7b1"

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
})


def compute_gini(wealths):
    total = sum(wealths)
    if total == 0:
        return 0.0
    n = len(wealths)
    s = sorted(wealths)
    numerator = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(s))
    return numerator / (n * total)


def parse_experiment_data(output_dir):
    p = Path(output_dir) / "data" / "agent_episodes.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")

    with open(p, "r") as f:
        rows = list(csv.DictReader(f))

    ep_worlds = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ep_worlds[int(r["episode"])][int(r["world"])].append(r)

    episodes = []
    wealth_list = []
    gini_list = []
    houses_list = []
    resources_mined_list = []
    resources_inv_list = []
    coins_list = []

    for ep in sorted(ep_worlds.keys()):
        worlds = ep_worlds[ep]
        ep_houses = 0.0
        ep_coins = 0.0
        ep_wood = 0.0
        ep_stone = 0.0
        ep_ginis = []

        for w, agents in worlds.items():
            coins = [float(a["coin"]) for a in agents]
            wealths = [float(a["coin"]) + 2.0 * float(a["wood"]) + 2.0 * float(a["stone"]) for a in agents]
            ep_houses += sum(float(a["houses"]) for a in agents)
            ep_coins += sum(coins)
            ep_wood += sum(float(a["wood"]) for a in agents)
            ep_stone += sum(float(a["stone"]) for a in agents)
            ep_ginis.append(compute_gini(wealths))

        total_mined = 2.0 * ep_houses + ep_wood + ep_stone
        total_wealth = ep_coins + 2.0 * ep_wood + 2.0 * ep_stone
        mean_gini = sum(ep_ginis) / len(ep_ginis) if ep_ginis else 0.0

        episodes.append(ep)
        wealth_list.append(total_wealth)
        gini_list.append(mean_gini)
        houses_list.append(ep_houses)
        resources_mined_list.append(total_mined)
        resources_inv_list.append(ep_wood + ep_stone)
        coins_list.append(ep_coins)

    return {
        "episodes": np.array(episodes),
        "wealth": np.array(wealth_list),
        "gini": np.array(gini_list),
        "houses": np.array(houses_list),
        "resources_mined": np.array(resources_mined_list),
        "resources_inv": np.array(resources_inv_list),
        "coins": np.array(coins_list),
    }


def rolling_avg(arr, window=3):
    return np.convolve(arr, np.ones(window)/window, mode="same")


def main():
    out_dir = Path("outputs/comparison_exp7_vs_exp8")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for Experiment 7 (Voting) and Experiment 8 (Central Planning)...")
    d7 = parse_experiment_data("outputs/experiment7_voting_uniform_fungible")
    d8 = parse_experiment_data("outputs/experiment8_normal_uniform_fungible")

    eps = d7["episodes"]
    n_eps = len(eps)

    # =========================================================================
    # 1. COMBINED 4-PANEL DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=250)
    fig.suptitle(
        "Macroeconomic Comparison Across Episodes: Democratic Voting vs. Central Planning\n(Fungible Resources, +40 Build Payment, +5 Resource Pickup)",
        fontweight="bold",
        fontsize=15,
        y=0.98,
    )

    # Panel 1: Wealth Generated
    ax1 = axes[0, 0]
    ax1.plot(eps, d7["wealth"], color=COLOR_EXP7, marker="o", markersize=4, linewidth=1.8, label="Exp 7: Democratic Voting", alpha=0.9)
    ax1.plot(eps, d8["wealth"], color=COLOR_EXP8, marker="s", markersize=4, linewidth=1.8, label="Exp 8: Central Planning", alpha=0.9)
    ax1.fill_between(eps, d7["wealth"], color=COLOR_EXP7, alpha=0.12)
    ax1.fill_between(eps, d8["wealth"], color=COLOR_EXP8, alpha=0.12)
    ax1.set_title("1. Wealth Generated per Episode (Coins + Resources)", fontweight="bold")
    ax1.set_xlabel("Episode (300 steps per world across 16 worlds)")
    ax1.set_ylabel("Total Wealth Value")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)
    tot_w7 = np.sum(d7["wealth"])
    tot_w8 = np.sum(d8["wealth"])
    ax1.text(0.03, 0.90, f"Total Wealth:\n  Exp 7: {tot_w7:,.0f}\n  Exp 8: {tot_w8:,.0f} (+{(tot_w7/tot_w8-1)*100:.1f}%)", 
             transform=ax1.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 2: Gini Coefficient
    ax2 = axes[0, 1]
    ax2.plot(eps, d7["gini"], color=COLOR_EXP7, marker="o", markersize=4, linewidth=1.8, label="Exp 7: Democratic Voting", alpha=0.9)
    ax2.plot(eps, d8["gini"], color=COLOR_EXP8, marker="s", markersize=4, linewidth=1.8, label="Exp 8: Central Planning", alpha=0.9)
    ax2.plot(eps, rolling_avg(d7["gini"], 5), color="#0b3c68", linestyle="--", linewidth=2.0, label="Exp 7 (5-Ep Trend)")
    ax2.plot(eps, rolling_avg(d8["gini"], 5), color="#781113", linestyle="--", linewidth=2.0, label="Exp 8 (5-Ep Trend)")
    ax2.set_title("2. Wealth Inequality (Mean Gini Coefficient)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Gini Coefficient [0 = Equality, 1 = Inequality]")
    ax2.set_ylim([0.60, 0.98])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.text(0.03, 0.12, f"Mean Gini:\n  Exp 7: {np.mean(d7['gini']):.4f}\n  Exp 8: {np.mean(d8['gini']):.4f}", 
             transform=ax2.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 3: Houses Built
    ax3 = axes[1, 0]
    ax3.plot(eps, d7["houses"], color=COLOR_EXP7, marker="o", markersize=4, linewidth=1.8, label="Exp 7: Democratic Voting", alpha=0.9)
    ax3.plot(eps, d8["houses"], color=COLOR_EXP8, marker="s", markersize=4, linewidth=1.8, label="Exp 8: Central Planning", alpha=0.9)
    ax3.fill_between(eps, d7["houses"], color=COLOR_EXP7, alpha=0.15)
    ax3.fill_between(eps, d8["houses"], color=COLOR_EXP8, alpha=0.15)
    ax3.set_title("3. Houses Built per Episode", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Houses Constructed")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.9)
    tot_h7 = np.sum(d7["houses"])
    tot_h8 = np.sum(d8["houses"])
    ax3.text(0.03, 0.85, f"Total Houses Built:\n  Exp 7: {tot_h7:,.0f} houses\n  Exp 8: {tot_h8:,.0f} houses\n  Advantage: +{(tot_h7/tot_h8-1)*100:.1f}%", 
             transform=ax3.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 4: Resources Accumulated / Mined
    ax4 = axes[1, 1]
    ax4.plot(eps, d7["resources_mined"], color=COLOR_EXP7, marker="o", markersize=4, linewidth=1.8, label="Exp 7: Total Mined (Wood + Stone)", alpha=0.9)
    ax4.plot(eps, d8["resources_mined"], color=COLOR_EXP8, marker="s", markersize=4, linewidth=1.8, label="Exp 8: Total Mined (Wood + Stone)", alpha=0.9)
    ax4.plot(eps, d7["resources_inv"], color="#0b3c68", linestyle=":", linewidth=1.6, label="Exp 7: Idle Inventory (Unbuilt)")
    ax4.plot(eps, d8["resources_inv"], color="#781113", linestyle=":", linewidth=1.6, label="Exp 8: Idle Inventory (Unbuilt)")
    ax4.set_title("4. Resources Accumulated & Utilization per Episode", fontweight="bold")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Resource Units (Wood + Stone)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    tot_m7 = np.sum(d7["resources_mined"])
    tot_m8 = np.sum(d8["resources_mined"])
    conv7 = (tot_h7 * 2.0 / tot_m7) * 100
    conv8 = (tot_h8 * 2.0 / tot_m8) * 100
    ax4.text(0.03, 0.12, f"Conversion Efficiency:\n  Exp 7: {conv7:.1f}% converted into housing\n  Exp 8: {conv8:.1f}% converted into housing", 
             transform=ax4.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    combined_path = out_dir / "combined_comparison_dashboard.png"
    fig.savefig(combined_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    # =========================================================================
    # 2. STANDALONE INDIVIDUAL GRAPHS
    # =========================================================================

    # Graph A: Wealth Generated
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(eps, d7["wealth"], color=COLOR_EXP7, marker="o", markersize=5, linewidth=2.2, label=f"Exp 7: Democratic Voting (Total: {tot_w7:,.0f})", alpha=0.9)
    ax.plot(eps, d8["wealth"], color=COLOR_EXP8, marker="s", markersize=5, linewidth=2.2, label=f"Exp 8: Central Planning (Total: {tot_w8:,.0f})", alpha=0.9)
    ax.fill_between(eps, d7["wealth"], color=COLOR_EXP7, alpha=0.15)
    ax.fill_between(eps, d8["wealth"], color=COLOR_EXP8, alpha=0.15)
    ax.plot(eps, rolling_avg(d7["wealth"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 7 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d8["wealth"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 8 (5-Ep Moving Avg)")
    ax.set_title("Wealth Generated per Episode: Democratic Voting vs. Central Planning", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode (300 timesteps per world across 16 parallel worlds)")
    ax.set_ylabel("Total Wealth Generated (Coins + Resource Valuation)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    g_wealth = out_dir / "wealth_generated_comparison.png"
    fig.savefig(g_wealth, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_wealth}")

    # Graph B: Gini Coefficient
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(eps, d7["gini"], color=COLOR_EXP7, marker="o", markersize=5, linewidth=2.0, label=f"Exp 7: Democratic Voting (Mean: {np.mean(d7['gini']):.4f})", alpha=0.85)
    ax.plot(eps, d8["gini"], color=COLOR_EXP8, marker="s", markersize=5, linewidth=2.0, label=f"Exp 8: Central Planning (Mean: {np.mean(d8['gini']):.4f})", alpha=0.85)
    ax.plot(eps, rolling_avg(d7["gini"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 7 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d8["gini"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 8 (5-Ep Moving Avg)")
    ax.set_title("Wealth Inequality Dynamics (Mean Gini Coefficient per Episode)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Gini Coefficient")
    ax.set_ylim([0.60, 0.98])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    g_gini = out_dir / "gini_inequality_comparison.png"
    fig.savefig(g_gini, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_gini}")

    # Graph C: Houses Built
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(eps, d7["houses"], color=COLOR_EXP7, marker="o", markersize=5, linewidth=2.2, label=f"Exp 7: Democratic Voting (Total: {tot_h7:,.0f})", alpha=0.9)
    ax.plot(eps, d8["houses"], color=COLOR_EXP8, marker="s", markersize=5, linewidth=2.2, label=f"Exp 8: Central Planning (Total: {tot_h8:,.0f})", alpha=0.9)
    ax.fill_between(eps, d7["houses"], color=COLOR_EXP7, alpha=0.15)
    ax.fill_between(eps, d8["houses"], color=COLOR_EXP8, alpha=0.15)
    ax.plot(eps, rolling_avg(d7["houses"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 7 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d8["houses"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 8 (5-Ep Moving Avg)")
    ax.set_title("Houses Built per Episode: Democratic Voting vs. Central Planning", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Houses Built")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    g_houses = out_dir / "houses_built_comparison.png"
    fig.savefig(g_houses, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_houses}")

    # Graph D: Resources Accumulated
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(eps, d7["resources_mined"], color=COLOR_EXP7, marker="o", markersize=5, linewidth=2.2, label=f"Exp 7: Total Resources Mined ({tot_m7:,.0f})", alpha=0.9)
    ax.plot(eps, d8["resources_mined"], color=COLOR_EXP8, marker="s", markersize=5, linewidth=2.2, label=f"Exp 8: Total Resources Mined ({tot_m8:,.0f})", alpha=0.9)
    ax.plot(eps, d7["resources_inv"], color="#0b3c68", linestyle=":", linewidth=2.0, label=f"Exp 7: Idle In-Inventory ({np.sum(d7['resources_inv']):,.0f})")
    ax.plot(eps, d8["resources_inv"], color="#781113", linestyle=":", linewidth=2.0, label=f"Exp 8: Idle In-Inventory ({np.sum(d8['resources_inv']):,.0f})")
    ax.set_title("Resources Accumulated & Utilization per Episode (Wood + Stone)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Resource Units")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    g_res = out_dir / "resources_accumulated_comparison.png"
    fig.savefig(g_res, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_res}")

    # =========================================================================
    # 3. CUMULATIVE GROWTH OVER TIME
    # =========================================================================
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15, 6), dpi=250)
    fig.suptitle("Cumulative Macroeconomic Trajectory (41 Episodes / 200,000 Global Steps)", fontweight="bold", fontsize=15)

    cum_h7 = np.cumsum(d7["houses"])
    cum_h8 = np.cumsum(d8["houses"])
    cum_w7 = np.cumsum(d7["wealth"])
    cum_w8 = np.cumsum(d8["wealth"])

    ax_c1.plot(eps, cum_h7, color=COLOR_EXP7, linewidth=2.8, label=f"Exp 7: Democratic Voting ({cum_h7[-1]:,.0f} houses)")
    ax_c1.plot(eps, cum_h8, color=COLOR_EXP8, linewidth=2.8, label=f"Exp 8: Central Planning ({cum_h8[-1]:,.0f} houses)")
    ax_c1.fill_between(eps, cum_h7, cum_h8, color=COLOR_EXP7, alpha=0.18, label=f"Democratic Surplus (+{cum_h7[-1]-cum_h8[-1]:,.0f} houses)")
    ax_c1.set_title("Cumulative Houses Built", fontweight="bold")
    ax_c1.set_xlabel("Episode")
    ax_c1.set_ylabel("Cumulative Houses")
    ax_c1.grid(True, linestyle="--", alpha=0.4)
    ax_c1.legend(loc="upper left", framealpha=0.9)

    ax_c2.plot(eps, cum_w7, color=COLOR_EXP7, linewidth=2.8, label=f"Exp 7: Democratic Voting ({cum_w7[-1]:,.0f} wealth)")
    ax_c2.plot(eps, cum_w8, color=COLOR_EXP8, linewidth=2.8, label=f"Exp 8: Central Planning ({cum_w8[-1]:,.0f} wealth)")
    ax_c2.fill_between(eps, cum_w7, cum_w8, color=COLOR_EXP7, alpha=0.18, label=f"Democratic Surplus (+{cum_w7[-1]-cum_w8[-1]:,.0f} wealth)")
    ax_c2.set_title("Cumulative Wealth Generated", fontweight="bold")
    ax_c2.set_xlabel("Episode")
    ax_c2.set_ylabel("Cumulative Wealth (Coins + Valuation)")
    ax_c2.grid(True, linestyle="--", alpha=0.4)
    ax_c2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    g_cum = out_dir / "cumulative_growth_comparison.png"
    fig.savefig(g_cum, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_cum}")

    # Save numeric summary JSON
    summary_data = {
        "episodes": int(n_eps),
        "Exp7_Democratic": {
            "total_houses": float(tot_h7),
            "total_wealth": float(tot_w7),
            "total_resources_mined": float(tot_m7),
            "conversion_efficiency": float(conv7),
            "mean_gini": float(np.mean(d7["gini"])),
            "per_episode_houses": d7["houses"].tolist(),
            "per_episode_wealth": d7["wealth"].tolist(),
            "per_episode_gini": d7["gini"].tolist(),
            "per_episode_resources_mined": d7["resources_mined"].tolist(),
        },
        "Exp8_CentralPlanning": {
            "total_houses": float(tot_h8),
            "total_wealth": float(tot_w8),
            "total_resources_mined": float(tot_m8),
            "conversion_efficiency": float(conv8),
            "mean_gini": float(np.mean(d8["gini"])),
            "per_episode_houses": d8["houses"].tolist(),
            "per_episode_wealth": d8["wealth"].tolist(),
            "per_episode_gini": d8["gini"].tolist(),
            "per_episode_resources_mined": d8["resources_mined"].tolist(),
        }
    }
    with open(out_dir / "comparison_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nAll comparative graphs and JSON metrics successfully generated!")


if __name__ == "__main__":
    main()

