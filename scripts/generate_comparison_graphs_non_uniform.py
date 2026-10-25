#!/usr/bin/env python3
"""Generate publication-grade comparative graphs between Experiment 9 (Voting Non-Uniform) and Experiment 10 (Central Planner Non-Uniform).

Compares:
1. Wealth Generated over each episode
2. Gini (Wealth Inequality) over each episode
3. Houses Built over each episode (strictly per-episode, non-cumulative)
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
COLOR_EXP9 = "#1f77b4"      # Democratic Voting (Blue)
COLOR_EXP9_FILL = "#a6c8e0"
COLOR_EXP10 = "#d62728"     # Central Planning (Red)
COLOR_EXP10_FILL = "#f5b7b1"

plt.rcParams.update({
    "font.family": "sans-serif",
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

        total_mined = ep_wood + ep_stone + (ep_houses * 2.0)
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
    out_dir = Path("outputs/comparison_exp9_vs_exp10")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for Experiment 9 (Voting Non-Uniform) and Experiment 10 (Planner Non-Uniform)...")
    d9 = parse_experiment_data("outputs/experiment9_voting_non_uniform_fungible")
    d10 = parse_experiment_data("outputs/experiment10_normal_non_uniform_fungible")

    # Match common episode length
    min_len = min(len(d9["episodes"]), len(d10["episodes"]))
    for k in d9:
        d9[k] = d9[k][:min_len]
    for k in d10:
        d10[k] = d10[k][:min_len]

    eps = d9["episodes"]
    n_eps = len(eps)

    tot_w9 = np.sum(d9["wealth"])
    tot_w10 = np.sum(d10["wealth"])
    tot_h9 = np.sum(d9["houses"])
    tot_h10 = np.sum(d10["houses"])
    tot_m9 = np.sum(d9["resources_mined"])
    tot_m10 = np.sum(d10["resources_mined"])
    conv9 = (tot_h9 * 2.0 / tot_m9) * 100 if tot_m9 > 0 else 0
    conv10 = (tot_h10 * 2.0 / tot_m10) * 100 if tot_m10 > 0 else 0

    # =========================================================================
    # 1. COMBINED 4-PANEL DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=250)
    fig.suptitle(
        "Macroeconomic Comparison Across Episodes: Non-Uniform Workers (Disparity)\nDemocratic Voting (Exp 9) vs. Central Planning (Exp 10) | Fungible Resources (+40/+5)",
        fontweight="bold",
        fontsize=15,
        y=0.98,
    )

    # Panel 1: Wealth Generated
    ax1 = axes[0, 0]
    ax1.plot(eps, d9["wealth"], color=COLOR_EXP9, marker="o", markersize=4, linewidth=1.8, label="Exp 9: Democratic Voting", alpha=0.9)
    ax1.plot(eps, d10["wealth"], color=COLOR_EXP10, marker="s", markersize=4, linewidth=1.8, label="Exp 10: Central Planning", alpha=0.9)
    ax1.fill_between(eps, d9["wealth"], color=COLOR_EXP9, alpha=0.12)
    ax1.fill_between(eps, d10["wealth"], color=COLOR_EXP10, alpha=0.12)
    ax1.set_title("1. Wealth Generated per Episode (Coins + Resources)", fontweight="bold")
    ax1.set_xlabel("Episode (300 steps per world across 16 worlds)")
    ax1.set_ylabel("Total Wealth Value")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.text(0.03, 0.90, f"Total Wealth:\n  Exp 9: {tot_w9:,.0f}\n  Exp 10: {tot_w10:,.0f} (+{(tot_w9/tot_w10-1)*100:.1f}%)", 
             transform=ax1.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 2: Gini Coefficient
    ax2 = axes[0, 1]
    ax2.plot(eps, d9["gini"], color=COLOR_EXP9, marker="o", markersize=4, linewidth=1.8, label="Exp 9: Democratic Voting", alpha=0.9)
    ax2.plot(eps, d10["gini"], color=COLOR_EXP10, marker="s", markersize=4, linewidth=1.8, label="Exp 10: Central Planning", alpha=0.9)
    ax2.plot(eps, rolling_avg(d9["gini"], 5), color="#0b3c68", linestyle="--", linewidth=2.0, label="Exp 9 (5-Ep Trend)")
    ax2.plot(eps, rolling_avg(d10["gini"], 5), color="#781113", linestyle="--", linewidth=2.0, label="Exp 10 (5-Ep Trend)")
    ax2.set_title("2. Wealth Inequality (Mean Gini Coefficient)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Gini Coefficient [0 = Equality, 1 = Inequality]")
    ax2.set_ylim([0.60, 0.98])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.text(0.03, 0.12, f"Mean Gini:\n  Exp 9: {np.mean(d9['gini']):.4f}\n  Exp 10: {np.mean(d10['gini']):.4f}", 
             transform=ax2.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 3: Houses Built
    ax3 = axes[1, 0]
    ax3.plot(eps, d9["houses"], color=COLOR_EXP9, marker="o", markersize=4, linewidth=1.8, label="Exp 9: Democratic Voting", alpha=0.9)
    ax3.plot(eps, d10["houses"], color=COLOR_EXP10, marker="s", markersize=4, linewidth=1.8, label="Exp 10: Central Planning", alpha=0.9)
    ax3.fill_between(eps, d9["houses"], color=COLOR_EXP9, alpha=0.15)
    ax3.fill_between(eps, d10["houses"], color=COLOR_EXP10, alpha=0.15)
    ax3.set_title("3. Houses Built per Episode (Non-Cumulative)", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Houses Constructed")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.9)
    ax3.text(0.03, 0.85, f"Total Houses Built:\n  Exp 9: {tot_h9:,.0f} houses\n  Exp 10: {tot_h10:,.0f} houses\n  Advantage: +{(tot_h9/tot_h10-1)*100:.1f}%", 
             transform=ax3.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.85))

    # Panel 4: Resources Accumulated / Mined
    ax4 = axes[1, 1]
    ax4.plot(eps, d9["resources_mined"], color=COLOR_EXP9, marker="o", markersize=4, linewidth=1.8, label="Exp 9: Total Mined (Wood + Stone)", alpha=0.9)
    ax4.plot(eps, d10["resources_mined"], color=COLOR_EXP10, marker="s", markersize=4, linewidth=1.8, label="Exp 10: Total Mined (Wood + Stone)", alpha=0.9)
    ax4.plot(eps, d9["resources_inv"], color="#0b3c68", linestyle=":", linewidth=1.6, label="Exp 9: Idle Inventory (Unbuilt)")
    ax4.plot(eps, d10["resources_inv"], color="#781113", linestyle=":", linewidth=1.6, label="Exp 10: Idle Inventory (Unbuilt)")
    ax4.set_title("4. Resources Accumulated & Utilization per Episode", fontweight="bold")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Resource Units (Wood + Stone)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    ax4.text(0.03, 0.12, f"Conversion Efficiency:\n  Exp 9: {conv9:.1f}% converted into housing\n  Exp 10: {conv10:.1f}% converted into housing", 
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
    ax.plot(eps, d9["wealth"], color=COLOR_EXP9, marker="o", markersize=5, linewidth=2.2, label=f"Exp 9: Democratic Voting (Total: {tot_w9:,.0f})", alpha=0.9)
    ax.plot(eps, d10["wealth"], color=COLOR_EXP10, marker="s", markersize=5, linewidth=2.2, label=f"Exp 10: Central Planning (Total: {tot_w10:,.0f})", alpha=0.9)
    ax.fill_between(eps, d9["wealth"], color=COLOR_EXP9, alpha=0.15)
    ax.fill_between(eps, d10["wealth"], color=COLOR_EXP10, alpha=0.15)
    ax.plot(eps, rolling_avg(d9["wealth"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 9 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d10["wealth"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 10 (5-Ep Moving Avg)")
    ax.set_title("Wealth Generated per Episode: Democratic Voting vs. Central Planning (Non-Uniform)", fontweight="bold", fontsize=14)
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
    ax.plot(eps, d9["gini"], color=COLOR_EXP9, marker="o", markersize=5, linewidth=2.0, label=f"Exp 9: Democratic Voting (Mean: {np.mean(d9['gini']):.4f})", alpha=0.85)
    ax.plot(eps, d10["gini"], color=COLOR_EXP10, marker="s", markersize=5, linewidth=2.0, label=f"Exp 10: Central Planning (Mean: {np.mean(d10['gini']):.4f})", alpha=0.85)
    ax.plot(eps, rolling_avg(d9["gini"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 9 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d10["gini"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 10 (5-Ep Moving Avg)")
    ax.set_title("Wealth Inequality Dynamics (Mean Gini Coefficient per Episode, Non-Uniform)", fontweight="bold", fontsize=14)
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
    ax.plot(eps, d9["houses"], color=COLOR_EXP9, marker="o", markersize=5, linewidth=2.2, label=f"Exp 9: Democratic Voting (Total: {tot_h9:,.0f})", alpha=0.9)
    ax.plot(eps, d10["houses"], color=COLOR_EXP10, marker="s", markersize=5, linewidth=2.2, label=f"Exp 10: Central Planning (Total: {tot_h10:,.0f})", alpha=0.9)
    ax.fill_between(eps, d9["houses"], color=COLOR_EXP9, alpha=0.15)
    ax.fill_between(eps, d10["houses"], color=COLOR_EXP10, alpha=0.15)
    ax.plot(eps, rolling_avg(d9["houses"], 5), color="#0b3c68", linestyle="--", linewidth=2.5, label="Exp 9 (5-Ep Moving Avg)")
    ax.plot(eps, rolling_avg(d10["houses"], 5), color="#781113", linestyle="--", linewidth=2.5, label="Exp 10 (5-Ep Moving Avg)")
    ax.set_title("Houses Built per Episode: Democratic Voting vs. Central Planning (Non-Uniform)", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Number of Houses Built")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    g_houses = out_dir / "houses_built_comparison.png"
    fig.savefig(g_houses, dpi=250)
    plt.close(fig)
    print(f"Saved: {g_houses}")

    # Graph D: Standalone Houses Built per Episode
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.plot(
        eps,
        d9["houses"],
        color=COLOR_EXP9,
        marker="o",
        markersize=6,
        linewidth=2.4,
        label=f"Exp 9: Democratic Voting (Mean: {np.mean(d9['houses']):.1f} houses/ep, Total: {tot_h9:,.0f})",
        zorder=4,
    )
    ax.plot(
        eps,
        d10["houses"],
        color=COLOR_EXP10,
        marker="s",
        markersize=6,
        linewidth=2.4,
        label=f"Exp 10: Central Planning (Mean: {np.mean(d10['houses']):.1f} houses/ep, Total: {tot_h10:,.0f})",
        zorder=4,
    )
    ax.fill_between(
        eps,
        d9["houses"],
        d10["houses"],
        where=(d9["houses"] >= d10["houses"]),
        interpolate=True,
        color=COLOR_EXP9,
        alpha=0.18,
        label="Democratic Production Advantage (Exp 9 > Exp 10)",
        zorder=2,
    )
    ax.fill_between(
        eps,
        d9["houses"],
        d10["houses"],
        where=(d9["houses"] < d10["houses"]),
        interpolate=True,
        color=COLOR_EXP10,
        alpha=0.18,
        label="Central Planner Production Advantage (Exp 10 > Exp 9)",
        zorder=2,
    )
    ax.set_title("Houses Built per Episode: Democratic Voting vs. Central Planning (Non-Uniform Workers)", fontweight="bold", fontsize=15)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Houses Constructed", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=11)
    
    # Text annotation
    adv_pct = (tot_h9 / tot_h10 - 1) * 100 if tot_h10 > 0 else 0
    ax.text(
        0.03, 0.82,
        f"Cumulative Summary ({n_eps} Episodes):\n"
        f"  • Exp 9 (Democratic Voting): {tot_h9:,.0f} houses\n"
        f"  • Exp 10 (Central Planning): {tot_h10:,.0f} houses\n"
        f"  • Net Democratic Surplus: +{tot_h9 - tot_h10:,.0f} houses (+{adv_pct:.1f}%)",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#0b3c68", linewidth=1.5, alpha=0.92),
        zorder=5,
    )
    fig.tight_layout()
    g_standalone = out_dir / "houses_built_per_episode_standalone.png"
    fig.savefig(g_standalone, dpi=300)
    plt.close(fig)
    print(f"Saved: {g_standalone}")

    # Graph E: Resources Accumulated
    fig, ax = plt.subplots(figsize=(10, 6), dpi=250)
    ax.plot(eps, d9["resources_mined"], color=COLOR_EXP9, marker="o", markersize=5, linewidth=2.2, label=f"Exp 9: Total Resources Mined ({tot_m9:,.0f})", alpha=0.9)
    ax.plot(eps, d10["resources_mined"], color=COLOR_EXP10, marker="s", markersize=5, linewidth=2.2, label=f"Exp 10: Total Resources Mined ({tot_m10:,.0f})", alpha=0.9)
    ax.plot(eps, d9["resources_inv"], color="#0b3c68", linestyle=":", linewidth=2.0, label=f"Exp 9: Idle In-Inventory ({np.sum(d9['resources_inv']):,.0f})")
    ax.plot(eps, d10["resources_inv"], color="#781113", linestyle=":", linewidth=2.0, label=f"Exp 10: Idle In-Inventory ({np.sum(d10['resources_inv']):,.0f})")
    ax.set_title("Resources Accumulated & Utilization per Episode (Wood + Stone, Non-Uniform)", fontweight="bold", fontsize=14)
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
    fig.suptitle(f"Cumulative Macroeconomic Trajectory ({n_eps} Episodes / 200,000 Steps, Non-Uniform)", fontweight="bold", fontsize=15)

    cum_h9 = np.cumsum(d9["houses"])
    cum_h10 = np.cumsum(d10["houses"])
    cum_w9 = np.cumsum(d9["wealth"])
    cum_w10 = np.cumsum(d10["wealth"])

    ax_c1.plot(eps, cum_h9, color=COLOR_EXP9, linewidth=2.8, label=f"Exp 9: Democratic Voting ({cum_h9[-1]:,.0f} houses)")
    ax_c1.plot(eps, cum_h10, color=COLOR_EXP10, linewidth=2.8, label=f"Exp 10: Central Planning ({cum_h10[-1]:,.0f} houses)")
    ax_c1.fill_between(eps, cum_h9, cum_h10, color=COLOR_EXP9, alpha=0.18, label=f"Democratic Surplus (+{cum_h9[-1]-cum_h10[-1]:,.0f} houses)")
    ax_c1.set_title("Cumulative Houses Built", fontweight="bold")
    ax_c1.set_xlabel("Episode")
    ax_c1.set_ylabel("Cumulative Houses")
    ax_c1.grid(True, linestyle="--", alpha=0.4)
    ax_c1.legend(loc="upper left", framealpha=0.9)

    ax_c2.plot(eps, cum_w9, color=COLOR_EXP9, linewidth=2.8, label=f"Exp 9: Democratic Voting ({cum_w9[-1]:,.0f} wealth)")
    ax_c2.plot(eps, cum_w10, color=COLOR_EXP10, linewidth=2.8, label=f"Exp 10: Central Planning ({cum_w10[-1]:,.0f} wealth)")
    ax_c2.fill_between(eps, cum_w9, cum_w10, color=COLOR_EXP9, alpha=0.18, label=f"Democratic Surplus (+{cum_w9[-1]-cum_w10[-1]:,.0f} wealth)")
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
        "Exp9_Democratic_NonUniform": {
            "total_houses": float(tot_h9),
            "total_wealth": float(tot_w9),
            "total_resources_mined": float(tot_m9),
            "conversion_efficiency": float(conv9),
            "mean_gini": float(np.mean(d9["gini"])),
            "per_episode_houses": d9["houses"].tolist(),
            "per_episode_wealth": d9["wealth"].tolist(),
            "per_episode_gini": d9["gini"].tolist(),
            "per_episode_resources_mined": d9["resources_mined"].tolist(),
        },
        "Exp10_CentralPlanning_NonUniform": {
            "total_houses": float(tot_h10),
            "total_wealth": float(tot_w10),
            "total_resources_mined": float(tot_m10),
            "conversion_efficiency": float(conv10),
            "mean_gini": float(np.mean(d10["gini"])),
            "per_episode_houses": d10["houses"].tolist(),
            "per_episode_wealth": d10["wealth"].tolist(),
            "per_episode_gini": d10["gini"].tolist(),
            "per_episode_resources_mined": d10["resources_mined"].tolist(),
        }
    }
    with open(out_dir / "comparison_metrics.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nAll comparative graphs and JSON metrics successfully generated!")


if __name__ == "__main__":
    main()

