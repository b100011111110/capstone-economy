#!/usr/bin/env python3
"""Generate publication-grade comprehensive comparative analysis across the full 8-experiment factorial matrix:

Factorial Dimensions:
1. Population: Equal (Uniform) vs. Disparate (Non-Uniform)
2. Governance: Central Planner (No Election) vs. Democratic Voting (Election)
3. Institutional Safeguard: Baseline Incentives vs. Anti-Crash Framework

Experiments:
- Exp 7:  Uniform Democracy Baseline
- Exp 8:  Uniform Planner Baseline
- Exp 9:  Non-Uniform Democracy Baseline
- Exp 10: Non-Uniform Planner Baseline
- Exp 11: Non-Uniform Democracy + Anti-Crash
- Exp 12: Uniform Democracy + Anti-Crash
- Exp 13: Uniform Planner + Anti-Crash
- Exp 14: Non-Uniform Planner + Anti-Crash
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "Exp 13": "#1b9e77",  # Dark Teal (Uniform Planner Anti-Crash)
    "Exp 11": "#2ca02c",  # Emerald Green (Non-Uniform Democracy Anti-Crash)
    "Exp 14": "#d95f02",  # Burnt Orange (Non-Uniform Planner Anti-Crash)
    "Exp 12": "#7570b3",  # Purple (Uniform Democracy Anti-Crash)
    "Exp 7":  "#1f77b4",  # Steel Blue (Uniform Democracy Baseline)
    "Exp 8":  "#ff7f0e",  # Orange (Uniform Planner Baseline)
    "Exp 9":  "#e377c2",  # Pink (Non-Uniform Democracy Baseline)
    "Exp 10": "#d62728",  # Red (Non-Uniform Planner Baseline)
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
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
        raise FileNotFoundError(f"Missing {p}")

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
        coins_list.append(ep_coins)

    return {
        "episodes": np.array(episodes),
        "wealth": np.array(wealth_list),
        "gini": np.array(gini_list),
        "houses": np.array(houses_list),
        "mined": np.array(resources_mined_list),
        "coins": np.array(coins_list),
    }


def main():
    experiments = [
        ("Exp 13", "outputs/experiment13_normal_uniform_anti_crash", "Exp 13: Uniform Planner + Anti-Crash", "P", "-", 2.5),
        ("Exp 11", "outputs/experiment11_voting_anti_crash", "Exp 11: Non-Uniform Democracy + Anti-Crash", "o", "-", 2.6),
        ("Exp 14", "outputs/experiment14_normal_non_uniform_anti_crash", "Exp 14: Non-Uniform Planner + Anti-Crash", "X", "-", 2.3),
        ("Exp 12", "outputs/experiment12_voting_uniform_anti_crash", "Exp 12: Uniform Democracy + Anti-Crash", "D", "-", 2.2),
        ("Exp 7",  "outputs/experiment7_voting_uniform_fungible", "Exp 7:  Uniform Democracy Baseline", "s", "--", 1.8),
        ("Exp 8",  "outputs/experiment8_normal_uniform_fungible", "Exp 8:  Uniform Planner Baseline", "^", ":", 1.6),
        ("Exp 9",  "outputs/experiment9_voting_non_uniform_fungible", "Exp 9:  Non-Uniform Democracy Baseline", "v", "--", 1.8),
        ("Exp 10", "outputs/experiment10_normal_non_uniform_fungible", "Exp 10: Non-Uniform Planner Baseline", "x", ":", 1.6),
    ]

    raw_data = {}
    metrics = {}
    for code, path, label, marker, linestyle, lw in experiments:
        d = parse_experiment_data(path)
        raw_data[code] = d
        tot_h = float(np.sum(d["houses"]))
        tot_w = float(np.sum(d["wealth"]))
        mean_h = float(np.mean(d["houses"]))
        mean_g = float(np.mean(d["gini"]))
        zero_h = int(np.sum(d["houses"] == 0))
        tot_mined = float(np.sum(d["mined"]))
        conv = (tot_h * 2.0 / tot_mined * 100.0) if tot_mined > 0 else 0.0
        metrics[code] = {
            "name": label,
            "path": path,
            "total_houses": tot_h,
            "mean_houses_per_ep": mean_h,
            "min_houses_per_ep": float(np.min(d["houses"])),
            "max_houses_per_ep": float(np.max(d["houses"])),
            "zero_house_episodes": zero_h,
            "total_wealth": tot_w,
            "mean_wealth_per_ep": float(np.mean(d["wealth"])),
            "mean_gini": mean_g,
            "final_gini": float(d["gini"][-1]),
            "resource_conversion_efficiency_pct": conv,
        }

    out_base = Path("outputs/factorial_8_comparison")
    out_base.mkdir(parents=True, exist_ok=True)

    with open(out_base / "factorial_8_summary_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {out_base / 'factorial_8_summary_metrics.json'}")

    # =========================================================================
    # 1. 4-PANEL COMPREHENSIVE OVERVIEW DASHBOARD (ALL 8 EXPERIMENTS)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 13), dpi=250)
    fig.suptitle("Full 8-Experiment Factorial Benchmark: Institutional Design Across Economies", fontweight="bold", fontsize=16, y=0.98)

    # Panel 1: Houses Built per Episode
    ax1 = axes[0, 0]
    for code, path, label, marker, linestyle, lw in experiments:
        d = raw_data[code]
        ax1.plot(d["episodes"], d["houses"], color=COLORS[code], marker=marker, markersize=4, linewidth=lw, linestyle=linestyle, label=f"{code}: {d['houses'].sum():,.0f} homes", alpha=0.9)
    ax1.set_title("A. Homes Built per Episode (Non-Cumulative)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Homes Built", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel 2: Total Wealth Generated per Episode
    ax2 = axes[0, 1]
    for code, path, label, marker, linestyle, lw in experiments:
        d = raw_data[code]
        ax2.plot(d["episodes"], d["wealth"], color=COLORS[code], marker=marker, markersize=4, linewidth=lw, linestyle=linestyle, label=f"{code}: {d['wealth'].sum():,.0f} wealth", alpha=0.9)
    ax2.set_title("B. Total Wealth Generated per Episode (Coin + Resources)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Total Wealth", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", framealpha=0.9, fontsize=9)

    # Panel 3: Gini Inequality Coefficient
    ax3 = axes[1, 0]
    for code, path, label, marker, linestyle, lw in experiments:
        d = raw_data[code]
        ax3.plot(d["episodes"], d["gini"], color=COLORS[code], marker=marker, markersize=4, linewidth=lw, linestyle=linestyle, label=f"{code} (Mean: {d['gini'].mean():.3f})", alpha=0.9)
    ax3.set_title("C. Wealth Inequality (Gini Coefficient)", fontweight="bold", fontsize=12)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.set_ylabel("Gini Index", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # Panel 4: Resource Conversion Efficiency (%)
    ax4 = axes[1, 1]
    exp_codes = [e[0] for e in experiments]
    conv_values = [metrics[c]["resource_conversion_efficiency_pct"] for c in exp_codes]
    bar_colors = [COLORS[c] for c in exp_codes]
    bars = ax4.bar(exp_codes, conv_values, color=bar_colors, alpha=0.85, edgecolor="#333333", linewidth=1.2)
    for bar, val in zip(bars, conv_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax4.set_title("D. Resource Conversion Efficiency (% Mined Resources Built into Homes)", fontweight="bold", fontsize=12)
    ax4.set_ylabel("Conversion Efficiency (%)", fontsize=11)
    ax4.set_ylim([0, 75])
    ax4.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    p1 = out_base / "dashboard_8_experiments_overview.png"
    fig.savefig(p1, dpi=250)
    plt.close(fig)
    print(f"Saved: {p1}")

    # =========================================================================
    # 2. PAIRWISE CENTRAL PLANNER ANALYSIS: WITH vs WITHOUT ANTI-CRASH
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), dpi=250)
    fig.suptitle("Central Planning (No Election): Impact of Anti-Crash Framework", fontweight="bold", fontsize=15)

    # Left: Uniform Central Planner (Exp 8 vs Exp 13)
    d8 = raw_data["Exp 8"]
    d13 = raw_data["Exp 13"]
    ax1.plot(d8["episodes"], d8["houses"], color=COLORS["Exp 8"], marker="^", linewidth=2.0, linestyle="--", label=f"Exp 8: Uniform Planner Baseline ({d8['houses'].sum():,.0f} homes)")
    ax1.plot(d13["episodes"], d13["houses"], color=COLORS["Exp 13"], marker="P", linewidth=2.5, linestyle="-", label=f"Exp 13: Uniform Planner + Anti-Crash ({d13['houses'].sum():,.0f} homes)")
    ax1.fill_between(d13["episodes"], d13["houses"], d8["houses"], color=COLORS["Exp 13"], alpha=0.15, label=f"Production Surge (+{d13['houses'].sum()-d8['houses'].sum():,.0f} homes, +{(d13['houses'].sum()/d8['houses'].sum()-1)*100:.1f}%)")
    ax1.set_title("Equal Population: Exp 8 vs Exp 13", fontweight="bold", fontsize=13)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Homes Built per Episode", fontsize=11)
    ax1.set_ylim([-10, 580])
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=10)

    # Right: Non-Uniform Central Planner (Exp 10 vs Exp 14)
    d10 = raw_data["Exp 10"]
    d14 = raw_data["Exp 14"]
    ax2.plot(d10["episodes"], d10["houses"], color=COLORS["Exp 10"], marker="x", linewidth=2.0, linestyle="--", label=f"Exp 10: Non-Uniform Planner Baseline ({d10['houses'].sum():,.0f} homes)")
    ax2.plot(d14["episodes"], d14["houses"], color=COLORS["Exp 14"], marker="X", linewidth=2.5, linestyle="-", label=f"Exp 14: Non-Uniform Planner + Anti-Crash ({d14['houses'].sum():,.0f} homes)")
    ax2.fill_between(d14["episodes"], d14["houses"], d10["houses"], color=COLORS["Exp 14"], alpha=0.15, label=f"Production Surge (+{d14['houses'].sum()-d10['houses'].sum():,.0f} homes, +{(d14['houses'].sum()/d10['houses'].sum()-1)*100:.1f}%)")
    ax2.set_title("Disparate Population: Exp 10 vs Exp 14", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Homes Built per Episode", fontsize=11)
    ax2.set_ylim([-10, 580])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p2 = out_base / "pairwise_planner_anticrash_comparison.png"
    fig.savefig(p2, dpi=250)
    plt.close(fig)
    print(f"Saved: {p2}")

    # =========================================================================
    # 3. DEMOCRACY VS PLANNER UNDER ANTI-CRASH
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7), dpi=250)
    fig.suptitle("Governance Comparison Under Anti-Crash: Democratic Voting vs. Central Planning", fontweight="bold", fontsize=15)

    # Left: Disparate Society (Exp 11 Democracy vs Exp 14 Planner)
    d11 = raw_data["Exp 11"]
    ax1.plot(d11["episodes"], d11["houses"], color=COLORS["Exp 11"], marker="o", linewidth=2.5, label=f"Exp 11: Democracy + Anti-Crash ({d11['houses'].sum():,.0f} homes)")
    ax1.plot(d14["episodes"], d14["houses"], color=COLORS["Exp 14"], marker="X", linewidth=2.2, linestyle="--", label=f"Exp 14: Planner + Anti-Crash ({d14['houses'].sum():,.0f} homes)")
    ax1.set_title("Disparate Population (Non-Uniform Workers)", fontweight="bold", fontsize=13)
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Homes Built per Episode", fontsize=11)
    ax1.set_ylim([-10, 580])
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=10)

    # Right: Equal Society (Exp 12 Democracy vs Exp 13 Planner)
    d12 = raw_data["Exp 12"]
    ax2.plot(d12["episodes"], d12["houses"], color=COLORS["Exp 12"], marker="D", linewidth=2.3, linestyle="--", label=f"Exp 12: Democracy + Anti-Crash ({d12['houses'].sum():,.0f} homes)")
    ax2.plot(d13["episodes"], d13["houses"], color=COLORS["Exp 13"], marker="P", linewidth=2.5, label=f"Exp 13: Planner + Anti-Crash ({d13['houses'].sum():,.0f} homes)")
    ax2.set_title("Equal Population (Uniform Workers)", fontweight="bold", fontsize=13)
    ax2.set_xlabel("Episode", fontsize=11)
    ax2.set_ylabel("Homes Built per Episode", fontsize=11)
    ax2.set_ylim([-10, 580])
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p3 = out_base / "democracy_vs_planner_anticrash_comparison.png"
    fig.savefig(p3, dpi=250)
    plt.close(fig)
    print(f"Saved: {p3}")


if __name__ == "__main__":
    main()

