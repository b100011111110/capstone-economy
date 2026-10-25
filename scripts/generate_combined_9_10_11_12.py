#!/usr/bin/env python3
"""Generate publication-grade unified comparative graphs combining Experiments 9, 10, 11, and 12 into a single unified graph.

Experiments:
- Experiment 9: Non-Uniform + Democratic Voting Baseline (Without Anti-Crash)
- Experiment 10: Non-Uniform + Central Planning Baseline (Without Anti-Crash)
- Experiment 11: Non-Uniform + Democratic Voting WITH Anti-Crash & Sociotropic Accountability
- Experiment 12: Uniform + Democratic Voting WITH Anti-Crash & Sociotropic Accountability

Outputs:
1. unified_houses_built_per_episode.png (The primary unified single graph requested by the user)
2. unified_4panel_macro_dashboard.png (Combined 4-panel macroeconomic dashboard)
3. unified_cumulative_wealth_and_houses.png (Cumulative trajectory comparison)
4. unified_metrics_summary.json (Complete numeric metrics)
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Distinct styling palette
COLOR_EXP11 = "#2ca02c"   # Emerald Green (Exp 11: Non-Uniform Anti-Crash)
COLOR_EXP12 = "#9467bd"   # Deep Purple (Exp 12: Uniform Anti-Crash)
COLOR_EXP9  = "#1f77b4"   # Steel Blue (Exp 9: Non-Uniform Voting Baseline)
COLOR_EXP10 = "#d62728"   # Crimson Red (Exp 10: Non-Uniform Planner Baseline)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
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
    out_dir = Path("outputs/combined_experiments_9_10_11_12")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing data for Experiments 9, 10, 11, and 12...")
    d9 = parse_experiment_data("outputs/experiment9_voting_non_uniform_fungible")
    d10 = parse_experiment_data("outputs/experiment10_normal_non_uniform_fungible")
    d11 = parse_experiment_data("outputs/experiment11_voting_anti_crash")
    d12 = parse_experiment_data("outputs/experiment12_voting_uniform_anti_crash")

    eps = d9["episodes"]
    n_eps = len(eps)

    tot_h9 = np.sum(d9["houses"])
    tot_h10 = np.sum(d10["houses"])
    tot_h11 = np.sum(d11["houses"])
    tot_h12 = np.sum(d12["houses"])

    tot_w9 = np.sum(d9["wealth"])
    tot_w10 = np.sum(d10["wealth"])
    tot_w11 = np.sum(d11["wealth"])
    tot_w12 = np.sum(d12["wealth"])

    # =========================================================================
    # 1. PRIMARY UNIFIED GRAPH: HOUSES BUILT PER EPISODE (STRICTLY NON-CUMULATIVE)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)

    # Plot lines with distinct styles and markers
    ax.plot(
        eps,
        d11["houses"],
        color=COLOR_EXP11,
        marker="o",
        markersize=6,
        linewidth=2.8,
        label=f"Exp 11: Non-Uniform + Democracy (WITH Anti-Crash)  [Total: {tot_h11:,.0f} houses, Mean: {np.mean(d11['houses']):.1f}/ep]",
        zorder=5,
    )
    ax.plot(
        eps,
        d12["houses"],
        color=COLOR_EXP12,
        marker="D",
        markersize=5,
        linewidth=2.4,
        label=f"Exp 12: Uniform + Democracy (WITH Anti-Crash)      [Total: {tot_h12:,.0f} houses, Mean: {np.mean(d12['houses']):.1f}/ep]",
        zorder=4,
    )
    ax.plot(
        eps,
        d9["houses"],
        color=COLOR_EXP9,
        marker="s",
        markersize=5,
        linewidth=2.0,
        label=f"Exp 9:  Non-Uniform + Democracy (WITHOUT Anti-Crash) [Total: {tot_h9:,.0f} houses, Mean: {np.mean(d9['houses']):.1f}/ep]",
        zorder=3,
    )
    ax.plot(
        eps,
        d10["houses"],
        color=COLOR_EXP10,
        marker="^",
        markersize=5,
        linewidth=2.0,
        label=f"Exp 10: Non-Uniform + Central Planner (Fixed Baseline) [Total: {tot_h10:,.0f} houses, Mean: {np.mean(d10['houses']):.1f}/ep]",
        zorder=2,
    )

    # Shaded area under Exp 11 highlighting the anti-crash production frontier
    ax.fill_between(eps, d11["houses"], color=COLOR_EXP11, alpha=0.12, zorder=1)

    # Key Milestone Annotations
    ax.annotate(
        "Exp 9 & 10 Collapse into 0-House Deadlock\n(Meltzer-Richard Tax Trap & Policy Freeze)",
        xy=(33, 0),
        xytext=(20, 75),
        arrowprops=dict(facecolor=COLOR_EXP10, shrink=0.08, width=1.5, headwidth=7),
        fontsize=10.5,
        fontweight="bold",
        color="#781113",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_EXP10, alpha=0.92),
        zorder=7,
    )

    ax.annotate(
        "Exp 11 All-Time Peak: 495 Houses\n(Capital Tax Credit + Public Housing Dividend)",
        xy=(12, d11["houses"][12]),
        xytext=(3, 520),
        arrowprops=dict(facecolor=COLOR_EXP11, shrink=0.08, width=1.8, headwidth=8),
        fontsize=10.5,
        fontweight="bold",
        color="#145a32",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor=COLOR_EXP11, alpha=0.92),
        zorder=7,
    )

    # Titles and axes
    ax.set_title(
        "Unified Macroeconomic Comparison: Houses Constructed per Episode (Non-Cumulative)\nEvaluating Institutional Design Across Uniform vs. Non-Uniform Populations (200,000 Global Steps)",
        fontweight="bold",
        fontsize=15,
        pad=15,
    )
    ax.set_xlabel("Episode (300 Timesteps across 16 Parallel Worlds)", fontsize=12)
    ax.set_ylabel("Houses Constructed per Episode", fontsize=12)
    ax.set_ylim([-15, 570])
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10.5)

    # Comprehensive summary callout box
    summary_text = (
        f"Macro Summary Across All 41 Episodes:\n"
        f"  • Exp 11 (Non-Uniform + Anti-Crash):  {tot_h11:,.0f} houses (Zero 0-house episodes!)\n"
        f"  • Exp 12 (Uniform + Anti-Crash):       {tot_h12:,.0f} houses (Sustained late production)\n"
        f"  • Exp 9  (Non-Uniform Baseline):       {tot_h9:,.0f} houses (8 episodes collapsed to 0)\n"
        f"  • Exp 10 (Central Planner Baseline):   {tot_h10:,.0f} houses (Persistent stagnation)\n"
        f"  ────────────────────────────────────────────\n"
        f"  ★ Anti-Crash Gain in Disparate Society (Exp 11 vs 9): +{tot_h11 - tot_h9:,.0f} houses (+{(tot_h11/tot_h9-1)*100:.1f}%)"
    )
    ax.text(
        0.025, 0.70,
        summary_text,
        transform=ax.transAxes,
        fontsize=10.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#145a32", linewidth=1.6, alpha=0.95),
        zorder=6,
    )

    fig.tight_layout()
    unified_houses_path = out_dir / "unified_houses_built_per_episode.png"
    fig.savefig(unified_houses_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {unified_houses_path}")

    # =========================================================================
    # 2. COMBINED 4-PANEL MACROECONOMIC DASHBOARD
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=250)
    fig.suptitle(
        "Macroeconomic Synthesis: Experiments 9, 10, 11, and 12\n(Non-Cumulative Metrics Across 41 Episodes / 200,000 Global Steps)",
        fontweight="bold",
        fontsize=16,
        y=0.98,
    )

    # Panel 1: Houses Built per Episode
    ax1 = axes[0, 0]
    ax1.plot(eps, d11["houses"], color=COLOR_EXP11, marker="o", markersize=4, linewidth=2.4, label=f"Exp 11: Non-Uniform Anti-Crash ({tot_h11:,.0f})")
    ax1.plot(eps, d12["houses"], color=COLOR_EXP12, marker="D", markersize=4, linewidth=2.0, label=f"Exp 12: Uniform Anti-Crash ({tot_h12:,.0f})")
    ax1.plot(eps, d9["houses"], color=COLOR_EXP9, marker="s", markersize=4, linewidth=1.8, label=f"Exp 9: Non-Uniform Baseline ({tot_h9:,.0f})")
    ax1.plot(eps, d10["houses"], color=COLOR_EXP10, marker="^", markersize=4, linewidth=1.8, label=f"Exp 10: Planner Baseline ({tot_h10:,.0f})")
    ax1.set_title("1. Houses Built per Episode (Strictly Non-Cumulative)", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Houses Constructed")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Panel 2: Total Wealth Generated per Episode
    ax2 = axes[0, 1]
    ax2.plot(eps, d11["wealth"], color=COLOR_EXP11, marker="o", markersize=4, linewidth=2.4, label=f"Exp 11: Non-Uniform Anti-Crash ({tot_w11:,.0f})")
    ax2.plot(eps, d12["wealth"], color=COLOR_EXP12, marker="D", markersize=4, linewidth=2.0, label=f"Exp 12: Uniform Anti-Crash ({tot_w12:,.0f})")
    ax2.plot(eps, d9["wealth"], color=COLOR_EXP9, marker="s", markersize=4, linewidth=1.8, label=f"Exp 9: Non-Uniform Baseline ({tot_w9:,.0f})")
    ax2.plot(eps, d10["wealth"], color=COLOR_EXP10, marker="^", markersize=4, linewidth=1.8, label=f"Exp 10: Planner Baseline ({tot_w10:,.0f})")
    ax2.set_title("2. Total Wealth Generated per Episode (Coins + Resources)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Wealth Value")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Panel 3: Wealth Inequality (Mean Gini Coefficient)
    ax3 = axes[1, 0]
    ax3.plot(eps, rolling_avg(d11["gini"], 3), color=COLOR_EXP11, linewidth=2.2, label=f"Exp 11 (Mean: {np.mean(d11['gini']):.4f})")
    ax3.plot(eps, rolling_avg(d12["gini"], 3), color=COLOR_EXP12, linewidth=2.0, label=f"Exp 12 (Mean: {np.mean(d12['gini']):.4f})")
    ax3.plot(eps, rolling_avg(d9["gini"], 3), color=COLOR_EXP9, linewidth=1.8, label=f"Exp 9 (Mean: {np.mean(d9['gini']):.4f})")
    ax3.plot(eps, rolling_avg(d10["gini"], 3), color=COLOR_EXP10, linewidth=1.8, label=f"Exp 10 (Mean: {np.mean(d10['gini']):.4f})")
    ax3.set_title("3. Wealth Inequality Dynamics (Mean Gini Coefficient, 3-Ep Moving Avg)", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Gini Coefficient [0 = Equal, 1 = Unequal]")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", framealpha=0.9, fontsize=9.5)

    # Panel 4: Resource Conversion Efficiency
    ax4 = axes[1, 1]
    conv9 = (d9["houses"] * 2.0 / np.maximum(1.0, d9["resources_mined"])) * 100
    conv10 = (d10["houses"] * 2.0 / np.maximum(1.0, d10["resources_mined"])) * 100
    conv11 = (d11["houses"] * 2.0 / np.maximum(1.0, d11["resources_mined"])) * 100
    conv12 = (d12["houses"] * 2.0 / np.maximum(1.0, d12["resources_mined"])) * 100

    ax4.plot(eps, rolling_avg(conv11, 3), color=COLOR_EXP11, linewidth=2.2, label=f"Exp 11: Anti-Crash (Mean: {np.mean(conv11):.1f}%)")
    ax4.plot(eps, rolling_avg(conv12, 3), color=COLOR_EXP12, linewidth=2.0, label=f"Exp 12: Anti-Crash (Mean: {np.mean(conv12):.1f}%)")
    ax4.plot(eps, rolling_avg(conv9, 3), color=COLOR_EXP9, linewidth=1.8, label=f"Exp 9: Baseline (Mean: {np.mean(conv9):.1f}%)")
    ax4.plot(eps, rolling_avg(conv10, 3), color=COLOR_EXP10, linewidth=1.8, label=f"Exp 10: Planner (Mean: {np.mean(conv10):.1f}%)")
    ax4.set_title("4. Industrial Resource Conversion Efficiency (% Mined Turned into Housing)", fontweight="bold")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Conversion Efficiency (%)")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    dashboard_path = out_dir / "unified_4panel_macro_dashboard.png"
    fig.savefig(dashboard_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {dashboard_path}")

    # =========================================================================
    # 3. CUMULATIVE TRAJECTORY
    # =========================================================================
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(16, 6), dpi=250)
    fig.suptitle("Cumulative Macroeconomic Trajectory (Episodes 0–40 / 200,000 Global Steps)", fontweight="bold", fontsize=15)

    ax_c1.plot(eps, np.cumsum(d11["houses"]), color=COLOR_EXP11, linewidth=3.0, label=f"Exp 11: Non-Uniform Anti-Crash ({tot_h11:,.0f})")
    ax_c1.plot(eps, np.cumsum(d12["houses"]), color=COLOR_EXP12, linewidth=2.5, label=f"Exp 12: Uniform Anti-Crash ({tot_h12:,.0f})")
    ax_c1.plot(eps, np.cumsum(d9["houses"]), color=COLOR_EXP9, linewidth=2.0, label=f"Exp 9: Non-Uniform Baseline ({tot_h9:,.0f})")
    ax_c1.plot(eps, np.cumsum(d10["houses"]), color=COLOR_EXP10, linewidth=2.0, label=f"Exp 10: Planner Baseline ({tot_h10:,.0f})")
    ax_c1.set_title("Cumulative Houses Built", fontweight="bold")
    ax_c1.set_xlabel("Episode")
    ax_c1.set_ylabel("Cumulative Houses")
    ax_c1.grid(True, linestyle="--", alpha=0.4)
    ax_c1.legend(loc="upper left", framealpha=0.9)

    ax_c2.plot(eps, np.cumsum(d11["wealth"]), color=COLOR_EXP11, linewidth=3.0, label=f"Exp 11: Non-Uniform Anti-Crash ({tot_w11:,.0f})")
    ax_c2.plot(eps, np.cumsum(d12["wealth"]), color=COLOR_EXP12, linewidth=2.5, label=f"Exp 12: Uniform Anti-Crash ({tot_w12:,.0f})")
    ax_c2.plot(eps, np.cumsum(d9["wealth"]), color=COLOR_EXP9, linewidth=2.0, label=f"Exp 9: Non-Uniform Baseline ({tot_w9:,.0f})")
    ax_c2.plot(eps, np.cumsum(d10["wealth"]), color=COLOR_EXP10, linewidth=2.0, label=f"Exp 10: Planner Baseline ({tot_w10:,.0f})")
    ax_c2.set_title("Cumulative Wealth Generated", fontweight="bold")
    ax_c2.set_xlabel("Episode")
    ax_c2.set_ylabel("Cumulative Wealth (Coins + Valuation)")
    ax_c2.grid(True, linestyle="--", alpha=0.4)
    ax_c2.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    cum_path = out_dir / "unified_cumulative_wealth_and_houses.png"
    fig.savefig(cum_path, dpi=250)
    plt.close(fig)
    print(f"Saved: {cum_path}")

    # 4. JSON SUMMARY
    summary_data = {
        "episodes": int(n_eps),
        "Exp9_NonUniform_Baseline": {
            "total_houses": float(tot_h9),
            "total_wealth": float(tot_w9),
            "mean_gini": float(np.mean(d9["gini"])),
            "per_episode_houses": d9["houses"].tolist(),
        },
        "Exp10_Planner_Baseline": {
            "total_houses": float(tot_h10),
            "total_wealth": float(tot_w10),
            "mean_gini": float(np.mean(d10["gini"])),
            "per_episode_houses": d10["houses"].tolist(),
        },
        "Exp11_NonUniform_AntiCrash": {
            "total_houses": float(tot_h11),
            "total_wealth": float(tot_w11),
            "mean_gini": float(np.mean(d11["gini"])),
            "per_episode_houses": d11["houses"].tolist(),
        },
        "Exp12_Uniform_AntiCrash": {
            "total_houses": float(tot_h12),
            "total_wealth": float(tot_w12),
            "mean_gini": float(np.mean(d12["gini"])),
            "per_episode_houses": d12["houses"].tolist(),
        }
    }
    with open(out_dir / "unified_metrics_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nAll unified comparative graphs and JSON metrics successfully generated!")


if __name__ == "__main__":
    main()

