#!/usr/bin/env python3
"""Generate a standalone, non-cumulative comparison graph: Houses Built per Episode.

Compares:
- Experiment 7: Democratic Voting (With Election)
- Experiment 8: Central Planning (Without Election)
Strictly per-episode (non-cumulative).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Styling
COLOR_VOTING = "#1f77b4"       # Deep Blue (Democratic Voting)
COLOR_PLANNER = "#d62728"      # Crimson Red (Central Planning)
COLOR_DIFF_FILL = "#d0e1f9"    # Soft blue fill for voting advantage

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})


def main():
    metrics_path = Path("outputs/comparison_exp7_vs_exp8/comparison_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    h_voting = np.array(data["Exp7_Democratic"]["per_episode_houses"])
    h_planner = np.array(data["Exp8_CentralPlanning"]["per_episode_houses"])
    episodes = np.arange(len(h_voting))

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    # 1. Main curves (strictly per-episode values)
    ax.plot(
        episodes,
        h_voting,
        color=COLOR_VOTING,
        marker="o",
        markersize=6,
        linewidth=2.4,
        label=f"Exp 7: Democratic Voting (Mean: {np.mean(h_voting):.1f} houses/ep)",
        zorder=4,
    )

    ax.plot(
        episodes,
        h_planner,
        color=COLOR_PLANNER,
        marker="s",
        markersize=6,
        linewidth=2.4,
        label=f"Exp 8: Central Planning (Mean: {np.mean(h_planner):.1f} houses/ep)",
        zorder=4,
    )

    # 2. Fill advantage region where Democratic Voting > Central Planning
    ax.fill_between(
        episodes,
        h_voting,
        h_planner,
        where=(h_voting >= h_planner),
        interpolate=True,
        color=COLOR_VOTING,
        alpha=0.18,
        label="Democratic Production Advantage (Exp 7 > Exp 8)",
        zorder=2,
    )

    # Fill region where Central Planning > Democratic Voting
    ax.fill_between(
        episodes,
        h_voting,
        h_planner,
        where=(h_voting < h_planner),
        interpolate=True,
        color=COLOR_PLANNER,
        alpha=0.18,
        label="Central Planning Advantage (Exp 8 > Exp 7)",
        zorder=2,
    )

    # 3. Annotations highlighting the key economic phenomena
    # Stagnation Trap in Central Planning (Episodes 14-29)
    ax.axvspan(14, 29, color="#ffebee", alpha=0.35, zorder=1)
    ax.annotate(
        "Central Planning Stagnation Trap\n(Episodes 14–29: Planner tax hikes\nsuppress builds to 0–2 houses)",
        xy=(21.5, 2),
        xytext=(21.5, 90),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#b71c1c",
        arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.9),
        zorder=5,
    )

    # Democratic Resilience
    ax.annotate(
        "Democratic Self-Correction\n(Voters reject high-tax incumbents;\nsustains 120–250 houses/ep)",
        xy=(20, 199),
        xytext=(13, 260),
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#0d47a1",
        arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#64b5f6", alpha=0.9),
        zorder=5,
    )

    # Labels and Titles
    ax.set_title(
        "Houses Built per Episode: Democratic Voting vs. Central Planning (Non-Cumulative)",
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Episode (300 timesteps per world across 16 parallel worlds)", labelpad=10)
    ax.set_ylabel("Houses Built in Episode", labelpad=10)
    ax.set_xlim([-0.8, len(episodes) - 0.2])
    ax.set_ylim([-10, 520])

    # Grid & Ticks
    ax.set_xticks(np.arange(0, len(episodes), 2))
    ax.grid(True, linestyle="--", alpha=0.45, zorder=1)

    # Legend
    ax.legend(loc="upper right", framealpha=0.92, edgecolor="#cccccc")

    # Add summary box
    summary_text = (
        f"Non-Cumulative Per-Episode Statistics:\n"
        f"  • Exp 7 (Voting):   Mean = {np.mean(h_voting):.1f} | Peak = {np.max(h_voting):.0f} | Ep >= 100 houses: {np.sum(h_voting >= 100)}/41 eps\n"
        f"  • Exp 8 (Planner):  Mean = {np.mean(h_planner):.1f} | Peak = {np.max(h_planner):.0f} | Ep >= 100 houses: {np.sum(h_planner >= 100)}/41 eps\n"
        f"  • Net Difference:   +{(np.mean(h_voting)/np.mean(h_planner)-1)*100:.1f}% more houses built per episode under Democracy"
    )
    ax.text(
        0.02,
        0.96,
        summary_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#b0bec5", alpha=0.95),
        zorder=5,
    )

    fig.tight_layout()
    out_file = Path("outputs/comparison_exp7_vs_exp8/houses_built_per_episode_standalone.png")
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Saved non-cumulative graph: {out_file}")


if __name__ == "__main__":
    main()

