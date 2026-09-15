import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_cluster_shift_plots(data_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    elections_path = data_dir / "elections.csv"
    loyalty_path = data_dir / "loyalty.csv"
    
    if not elections_path.exists():
        print(f"Error: {elections_path} not found.")
        return

    df_e = pd.read_csv(elections_path)
    df_l = pd.read_csv(loyalty_path) if loyalty_path.exists() else pd.DataFrame()
    
    print(f"Loaded {len(df_e)} elections and {len(df_l)} loyalty records.")

    # Sort elections by global_step
    df_e = df_e.sort_values("global_step").reset_index(drop=True)
    
    # Parse overall vote shares
    raw_shares = [json.loads(s) for s in df_e["vote_shares"]]
    shares_df = pd.DataFrame(raw_shares, columns=["Red (Leader 0)", "Green (Leader 1)", "Blue (Leader 2)"])
    shares_df["global_step"] = df_e["global_step"]

    # Parse cluster-specific vote shares
    # cluster_vote_shares: list of 3 clusters x 3 leaders
    c0_shares, c1_shares, c2_shares = [], [], []
    valid_mask = []
    for s in df_e["cluster_vote_shares"]:
        try:
            arr = json.loads(s)
            if len(arr) == 3 and len(arr[0]) == 3:
                c0_shares.append(arr[0])
                c1_shares.append(arr[1])
                c2_shares.append(arr[2])
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        except Exception:
            valid_mask.append(False)

    c0_df = pd.DataFrame(c0_shares, columns=["Red (Leader 0)", "Green (Leader 1)", "Blue (Leader 2)"])
    c1_df = pd.DataFrame(c1_shares, columns=["Red (Leader 0)", "Green (Leader 1)", "Blue (Leader 2)"])
    c2_df = pd.DataFrame(c2_shares, columns=["Red (Leader 0)", "Green (Leader 1)", "Blue (Leader 2)"])
    
    steps_valid = df_e["global_step"][valid_mask].reset_index(drop=True)
    c0_df["global_step"] = steps_valid
    c1_df["global_step"] = steps_valid
    c2_df["global_step"] = steps_valid

    # Color Palette: Red, Green, Blue
    COLOR_RED = "#e41a1c"    # Leader 0 (Centrist Plurality)
    COLOR_GREEN = "#4daf4a"  # Leader 1 (Fiscal Conservative / Capital)
    COLOR_BLUE = "#377eb8"   # Leader 2 (Progressive Welfare)

    # Style settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), sharex=True)
    fig.suptitle("Dynamic Voter Cluster Shift & Electoral Polarization (Red vs Green vs Blue)", fontsize=16, fontweight="bold", y=0.98)

    roll_window = 35

    # 1. Macro Overall Vote Shift
    ax = axes[0, 0]
    for col, color, label in [
        ("Red (Leader 0)", COLOR_RED, "Red: Leader 0 (Broad Coalition)"),
        ("Green (Leader 1)", COLOR_GREEN, "Green: Leader 1 (Fiscal Conservative)"),
        ("Blue (Leader 2)", COLOR_BLUE, "Blue: Leader 2 (Progressive Welfare)"),
    ]:
        rolled = shares_df[col].rolling(roll_window, min_periods=1).mean()
        ax.plot(shares_df["global_step"], rolled, color=color, label=label, linewidth=2.5)

    ax.axhline(0.333, color="gray", linestyle=":", alpha=0.7, label="Equal Split (33%)")
    ax.set_title("1. Overall Population Vote Share Shifts Over Time", fontweight="bold", fontsize=13)
    ax.set_ylabel("Vote Share (%)", fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 2. Dynamic Cluster 0 Shifts (Resource Gatherers / Lower Income)
    ax = axes[0, 1]
    r0 = c0_df["Red (Leader 0)"].rolling(roll_window, min_periods=1).mean()
    g0 = c0_df["Green (Leader 1)"].rolling(roll_window, min_periods=1).mean()
    b0 = c0_df["Blue (Leader 2)"].rolling(roll_window, min_periods=1).mean()
    
    # Normalize rolling sum to 100%
    tot0 = r0 + g0 + b0 + 1e-8
    r0_norm, g0_norm, b0_norm = r0 / tot0, g0 / tot0, b0 / tot0

    ax.stackplot(
        c0_df["global_step"],
        r0_norm, g0_norm, b0_norm,
        labels=["Red: Leader 0", "Green: Leader 1", "Blue: Leader 2"],
        colors=[COLOR_RED, COLOR_GREEN, COLOR_BLUE],
        alpha=0.75,
    )
    ax.set_title("2. Dynamic Cluster 0 (Resource Gatherers): Vote Shift Stream", fontweight="bold", fontsize=13)
    ax.set_ylabel("Cluster Vote Composition", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 3. Dynamic Cluster 1 Shifts (Middle Artisans)
    ax = axes[1, 0]
    r1 = c1_df["Red (Leader 0)"].rolling(roll_window, min_periods=1).mean()
    g1 = c1_df["Green (Leader 1)"].rolling(roll_window, min_periods=1).mean()
    b1 = c1_df["Blue (Leader 2)"].rolling(roll_window, min_periods=1).mean()
    tot1 = r1 + g1 + b1 + 1e-8
    r1_norm, g1_norm, b1_norm = r1 / tot1, g1 / tot1, b1 / tot1

    ax.stackplot(
        c1_df["global_step"],
        r1_norm, g1_norm, b1_norm,
        labels=["Red: Leader 0", "Green: Leader 1", "Blue: Leader 2"],
        colors=[COLOR_RED, COLOR_GREEN, COLOR_BLUE],
        alpha=0.75,
    )
    ax.set_title("3. Dynamic Cluster 1 (Middle Artisans): Vote Shift Stream", fontweight="bold", fontsize=13)
    ax.set_xlabel("Global Step (100k Training Horizon)", fontsize=11)
    ax.set_ylabel("Cluster Vote Composition", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 4. Dynamic Cluster 2 Shifts (Capital Accumulators / Elite Builders)
    ax = axes[1, 1]
    r2 = c2_df["Red (Leader 0)"].rolling(roll_window, min_periods=1).mean()
    g2 = c2_df["Green (Leader 1)"].rolling(roll_window, min_periods=1).mean()
    b2 = c2_df["Blue (Leader 2)"].rolling(roll_window, min_periods=1).mean()
    tot2 = r2 + g2 + b2 + 1e-8
    r2_norm, g2_norm, b2_norm = r2 / tot2, g2 / tot2, b2 / tot2

    ax.stackplot(
        c2_df["global_step"],
        r2_norm, g2_norm, b2_norm,
        labels=["Red: Leader 0", "Green: Leader 1", "Blue: Leader 2"],
        colors=[COLOR_RED, COLOR_GREEN, COLOR_BLUE],
        alpha=0.75,
    )
    ax.set_title("4. Dynamic Cluster 2 (Capital Builders): Vote Shift Stream", fontweight="bold", fontsize=13)
    ax.set_xlabel("Global Step (100k Training Horizon)", fontsize=11)
    ax.set_ylabel("Cluster Vote Composition", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    out_file = output_dir / "voter_cluster_shifts.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_file}")

    # ==========================================
    # 2. Phase-Based Voter Coalitions Summary Plot
    # ==========================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig2.suptitle("Electoral Coalitions by Training Phase (Red vs Green vs Blue)", fontsize=15, fontweight="bold", y=1.02)
    
    # Divide into 3 chronological phases
    df_e["phase"] = pd.cut(df_e["global_step"], bins=[0, 33333, 66666, 100000], labels=["Early Phase (0-33k)", "Mid Phase (33-66k)", "Late Convergence (66-100k)"])
    
    phases = ["Early Phase (0-33k)", "Mid Phase (33-66k)", "Late Convergence (66-100k)"]
    for idx, phase in enumerate(phases):
        p_df = df_e[df_e["phase"] == phase]
        if p_df.empty:
            continue
        p_c_shares = [json.loads(s) for s in p_df["cluster_vote_shares"] if s != "[]"]
        if not p_c_shares:
            continue
        p_arr = np.array(p_c_shares) # (N, 3 clusters, 3 leaders)
        p_mean = np.mean(p_arr, axis=0) # (3 clusters, 3 leaders)
        
        ax = axes2[idx]
        x = np.arange(3)
        width = 0.25
        
        ax.bar(x - width, p_mean[:, 0] * 100, width, label="Red (Leader 0)", color=COLOR_RED, alpha=0.85)
        ax.bar(x, p_mean[:, 1] * 100, width, label="Green (Leader 1)", color=COLOR_GREEN, alpha=0.85)
        ax.bar(x + width, p_mean[:, 2] * 100, width, label="Blue (Leader 2)", color=COLOR_BLUE, alpha=0.85)
        
        ax.set_title(phase, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(["Cluster 0\n(Gatherers)", "Cluster 1\n(Artisans)", "Cluster 2\n(Builders)"])
        if idx == 0:
            ax.set_ylabel("Vote Share (%)", fontsize=11)
        ax.set_ylim(0, 60)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig2.tight_layout()
    out_file2 = output_dir / "cluster_electoral_phases.png"
    fig2.savefig(out_file2, dpi=200)
    plt.close(fig2)
    print(f"Saved: {out_file2}")


def main():
    parser = argparse.ArgumentParser(description="Generate cluster and vote shift plots")
    parser.add_argument("--data-dir", default="outputs/3leaders/data", help="Path to input data directory")
    parser.add_argument("--output-dir", default="outputs/3leaders/plots", help="Path to output plot directory")
    args = parser.parse_args()
    
    generate_cluster_shift_plots(Path(args.data_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()

