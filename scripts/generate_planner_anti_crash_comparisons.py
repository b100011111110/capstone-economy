#!/usr/bin/env python3
"""Generate pairwise comparative directories:
1. outputs/comparison_exp8_vs_exp13 (Uniform Planner: Baseline vs Anti-Crash)
2. outputs/comparison_exp10_vs_exp14 (Non-Uniform Planner: Baseline vs Anti-Crash)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from generate_all_8_factorial_comparison import parse_experiment_data, COLORS

def generate_pairwise(exp_base_code, exp_ac_code, base_path, ac_path, title, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    d_base = parse_experiment_data(base_path)
    d_ac = parse_experiment_data(ac_path)
    
    eps = d_base["episodes"]
    c_base = COLORS[exp_base_code]
    c_ac = COLORS[exp_ac_code]
    
    tot_base = np.sum(d_base["houses"])
    tot_ac = np.sum(d_ac["houses"])
    gain = tot_ac - tot_base
    pct = (tot_ac / tot_base - 1.0) * 100.0
    
    # 1. 4-Panel Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=250)
    fig.suptitle(title, fontweight="bold", fontsize=15, y=0.98)
    
    # Houses
    ax1 = axes[0, 0]
    ax1.plot(eps, d_base["houses"], color=c_base, marker="s", linewidth=1.8, linestyle="--", label=f"{exp_base_code} Baseline ({tot_base:,.0f} homes)")
    ax1.plot(eps, d_ac["houses"], color=c_ac, marker="o", linewidth=2.4, label=f"{exp_ac_code} Anti-Crash ({tot_ac:,.0f} homes)")
    ax1.fill_between(eps, d_ac["houses"], d_base["houses"], color=c_ac, alpha=0.15, label=f"Surplus (+{gain:,.0f} homes, +{pct:.1f}%)")
    ax1.set_title("A. Non-Cumulative Houses Built per Episode", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Houses Built")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", framealpha=0.9)
    
    # Cumulative Growth
    ax2 = axes[0, 1]
    ax2.plot(eps, np.cumsum(d_base["houses"]), color=c_base, linewidth=2.0, linestyle="--", label=f"{exp_base_code} Baseline")
    ax2.plot(eps, np.cumsum(d_ac["houses"]), color=c_ac, linewidth=2.5, label=f"{exp_ac_code} Anti-Crash")
    ax2.set_title("B. Cumulative Production Frontier", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Cumulative Houses")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper left", framealpha=0.9)
    
    # Wealth
    ax3 = axes[1, 0]
    ax3.plot(eps, d_base["wealth"], color=c_base, linewidth=1.8, linestyle="--", label=f"{exp_base_code} Baseline ({d_base['wealth'].sum():,.0f})")
    ax3.plot(eps, d_ac["wealth"], color=c_ac, linewidth=2.2, label=f"{exp_ac_code} Anti-Crash ({d_ac['wealth'].sum():,.0f})")
    ax3.set_title("C. Wealth Generation per Episode", fontweight="bold")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Wealth")
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="lower right", framealpha=0.9)
    
    # Gini
    ax4 = axes[1, 1]
    ax4.plot(eps, d_base["gini"], color=c_base, linewidth=1.8, linestyle="--", label=f"{exp_base_code} Baseline (Mean: {d_base['gini'].mean():.3f})")
    ax4.plot(eps, d_ac["gini"], color=c_ac, linewidth=2.2, label=f"{exp_ac_code} Anti-Crash (Mean: {d_ac['gini'].mean():.3f})")
    ax4.set_title("D. Wealth Gini Coefficient", fontweight="bold")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Gini Index")
    ax4.grid(True, linestyle="--", alpha=0.4)
    ax4.legend(loc="upper right", framealpha=0.9)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / "combined_comparison_dashboard.png", dpi=250)
    plt.close(fig)
    
    # 2. Standalone Houses
    fig, ax = plt.subplots(figsize=(14, 7), dpi=250)
    ax.plot(eps, d_base["houses"], color=c_base, marker="s", linewidth=2.0, linestyle="--", label=f"{exp_base_code} Baseline ({tot_base:,.0f} homes)")
    ax.plot(eps, d_ac["houses"], color=c_ac, marker="o", linewidth=2.5, label=f"{exp_ac_code} Anti-Crash ({tot_ac:,.0f} homes)")
    ax.fill_between(eps, d_ac["houses"], d_base["houses"], color=c_ac, alpha=0.15, label=f"Anti-Crash Surplus (+{gain:,.0f} homes, +{pct:.1f}%)")
    ax.set_title(f"Houses Built Across Episodes: {exp_base_code} vs {exp_ac_code}", fontweight="bold", fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Houses Built per Episode")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_dir / "houses_built_per_episode_standalone.png", dpi=250)
    plt.close(fig)
    
    # Save comparison metrics
    summary = {
        "baseline_experiment": exp_base_code,
        "anti_crash_experiment": exp_ac_code,
        "baseline_houses": tot_base,
        "anti_crash_houses": tot_ac,
        "houses_gain": gain,
        "houses_pct_gain": pct,
        "baseline_wealth": float(np.sum(d_base["wealth"])),
        "anti_crash_wealth": float(np.sum(d_ac["wealth"])),
        "baseline_zero_house_episodes": int(np.sum(d_base["houses"] == 0)),
        "anti_crash_zero_house_episodes": int(np.sum(d_ac["houses"] == 0)),
    }
    with open(out_dir / "comparison_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pairwise comparison in: {out_dir}")

def main():
    generate_pairwise(
        "Exp 8", "Exp 13",
        "outputs/experiment8_normal_uniform_fungible",
        "outputs/experiment13_normal_uniform_anti_crash",
        "Uniform Population Central Planner: Exp 8 Baseline vs. Exp 13 Anti-Crash",
        "outputs/comparison_exp8_vs_exp13"
    )
    generate_pairwise(
        "Exp 10", "Exp 14",
        "outputs/experiment10_normal_non_uniform_fungible",
        "outputs/experiment14_normal_non_uniform_anti_crash",
        "Non-Uniform Population Central Planner: Exp 10 Baseline vs. Exp 14 Anti-Crash",
        "outputs/comparison_exp10_vs_exp14"
    )

if __name__ == "__main__":
    main()

