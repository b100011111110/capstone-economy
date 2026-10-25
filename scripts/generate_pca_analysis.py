#!/usr/bin/env python3
"""PCA Clustering and Centroid Extraction for Political-Economic Agent Analysis.
Pure NumPy and Matplotlib implementation (zero external dependencies).
Generates 3D and 2D PCA plots with centroids for Simulation 3 and Simulation 4.
"""

import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

COLOR_RED = "#E41A1C"     # Leader 0 (Centrist Coalition)
COLOR_GREEN = "#4DAF4A"   # Leader 1 (Growth / Low-Tax)
COLOR_BLUE = "#377EB8"    # Leader 2 (Progressive Welfare)
CLUSTER_COLORS = [COLOR_RED, COLOR_GREEN, COLOR_BLUE]


class NumpyPCA:
    """Pure NumPy Principal Component Analysis."""
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.components_ = None
        self.explained_variance_ratio_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std
        cov = np.cov(X_norm, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(0.0, eigenvalues[idx])
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components].T
        total_var = float(np.sum(eigenvalues) + 1e-8)
        self.explained_variance_ratio_ = (eigenvalues[:self.n_components] / total_var).astype(np.float32)
        return X_norm @ self.components_.T

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean) / self.std
        return X_norm @ self.components_.T


def numpy_kmeans(X: np.ndarray, k: int = 3, max_iters: int = 50, seed: int = 42):
    """Pure NumPy K-Means clustering algorithm."""
    np.random.seed(seed)
    idx = np.random.choice(len(X), size=k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max_iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(k):
            if np.sum(labels == c) > 0:
                centroids[c] = np.mean(X[labels == c], axis=0)
    return labels, centroids


def load_agent_episodes(csv_path: str):
    with open(csv_path, "r") as f:
        reader = list(csv.DictReader(f))
    has_eff = "efficiency" in reader[0]
    
    feat_names = ["Coin", "Wood", "Stone", "Labor", "Reward", "Loyalty L0", "Loyalty L1", "Loyalty L2"]
    if has_eff:
        feat_names.append("Efficiency")
        
    data = []
    for r in reader:
        row = [
            float(r.get("coin", 0.0)),
            float(r.get("wood", 0.0)),
            float(r.get("stone", 0.0)),
            float(r.get("labor", 0.0)),
            float(r.get("cumulative_reward", 0.0)),
            float(r.get("loyalty_leader_0", 0.0)),
            float(r.get("loyalty_leader_1", 0.0)),
            float(r.get("loyalty_leader_2", 0.0)),
        ]
        if has_eff:
            row.append(float(r.get("efficiency", 0.0)))
        data.append(row)
        
    return np.asarray(data, dtype=np.float32), feat_names


def analyze_and_plot(exp_dir: str, title_suffix: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(exp_dir) / "data" / "agent_episodes.csv"
    
    print(f"\n=======================================================")
    print(f"Running PCA & Centroid Analysis: {exp_dir}")
    print(f"=======================================================")
    
    X, feat_names = load_agent_episodes(str(csv_path))
    print(f"Loaded {len(X)} agent-episode observations across {len(feat_names)} dimensions.")
    
    # Standardize
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-6
    X_norm = (X - X_mean) / X_std
    
    # PCA
    pca = NumpyPCA(n_components=3)
    X_pca = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_
    print(f"Explained Variance: PC1={var_exp[0]*100:.2f}%, PC2={var_exp[1]*100:.2f}%, PC3={var_exp[2]*100:.2f}% (Total={np.sum(var_exp)*100:.2f}%)")
    
    # K-Means clustering (k=3)
    cluster_labels, norm_centroids = numpy_kmeans(X_norm, k=3, max_iters=50, seed=42)
    
    # Calculate Centroids in PCA space and in original feature space
    pca_centroids = np.zeros((3, 3))
    orig_centroids = np.zeros((3, len(feat_names)))
    
    print("\n--- CLUSTER CENTROIDS (Original Feature Space) ---")
    centroid_summary = {}
    for c in range(3):
        mask = cluster_labels == c
        pca_centroids[c] = np.mean(X_pca[mask], axis=0)
        orig_centroids[c] = np.mean(X[mask], axis=0)
        n_c = int(np.sum(mask))
        pct = float(n_c / len(X) * 100)
        
        c_dict = {feat_names[i]: float(orig_centroids[c, i]) for i in range(len(feat_names))}
        c_dict["Size"] = n_c
        c_dict["Pct"] = pct
        c_dict["PCA_Coords"] = [float(pca_centroids[c, 0]), float(pca_centroids[c, 1]), float(pca_centroids[c, 2])]
        centroid_summary[f"Cluster_{c}"] = c_dict
        
        print(f"Cluster {c} ({n_c} instances, {pct:.1f}%):")
        print(f"  PCA Coordinates: PC1={pca_centroids[c,0]:.3f}, PC2={pca_centroids[c,1]:.3f}, PC3={pca_centroids[c,2]:.3f}")
        for fn, val in zip(feat_names, orig_centroids[c]):
            print(f"    {fn:15s}: {val:8.3f}")
            
    with open(output_dir / "pca_centroids.json", "w") as f:
        json.dump(centroid_summary, f, indent=2)
        
    print("\n--- PCA Feature Loadings ---")
    loadings = pca.components_  # (3, n_features)
    for i, name in enumerate(feat_names):
        print(f"  {name:15s}: PC1={loadings[0, i]:+.3f}, PC2={loadings[1, i]:+.3f}, PC3={loadings[2, i]:+.3f}")
        
    # ==========================================
    # 1. 3D PCA PLOT
    # ==========================================
    fig3d = plt.figure(figsize=(12, 9))
    ax3d = fig3d.add_subplot(111, projection="3d")
    
    sample_size = min(3000, len(X_pca))
    sub_idx = np.random.choice(len(X_pca), size=sample_size, replace=False)
    
    for c in range(3):
        c_mask = cluster_labels[sub_idx] == c
        ax3d.scatter(
            X_pca[sub_idx][c_mask, 0],
            X_pca[sub_idx][c_mask, 1],
            X_pca[sub_idx][c_mask, 2],
            c=CLUSTER_COLORS[c],
            alpha=0.45,
            s=22,
            label=f"Cluster {c} ({np.sum(cluster_labels == c)/len(X)*100:.1f}%)"
        )
        # Plot Centroid
        ax3d.scatter(
            pca_centroids[c, 0],
            pca_centroids[c, 1],
            pca_centroids[c, 2],
            c=CLUSTER_COLORS[c],
            edgecolors="black",
            s=350,
            marker="*",
            linewidth=2.0,
            label=f"Centroid {c} (★)"
        )
        ax3d.text(
            pca_centroids[c, 0],
            pca_centroids[c, 1],
            pca_centroids[c, 2] + 0.35,
            f"C{c}",
            color="black",
            fontweight="bold",
            fontsize=12,
            ha="center"
        )
        
    ax3d.set_title(f"3D PCA Agent State Space & Cluster Centroids\n{title_suffix}", fontsize=14, fontweight="bold", pad=15)
    ax3d.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% Variance)", fontsize=11, labelpad=8)
    ax3d.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% Variance)", fontsize=11, labelpad=8)
    ax3d.set_zlabel(f"PC3 ({var_exp[2]*100:.1f}% Variance)", fontsize=11, labelpad=8)
    ax3d.view_init(elev=28, azim=125)
    ax3d.grid(True, alpha=0.3)
    ax3d.legend(loc="upper left", bbox_to_anchor=(0.02, 0.95), fontsize=10, framealpha=0.9)
    
    fig3d.tight_layout()
    p3d_out = output_dir / "pca_clusters_3d.png"
    fig3d.savefig(p3d_out, dpi=200)
    plt.close(fig3d)
    print(f"Saved: {p3d_out}")
    
    # ==========================================
    # 2. 2D PCA BIPLOT (PC1 vs PC2 & PC1 vs PC3)
    # ==========================================
    fig2d, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig2d.suptitle(f"2D PCA Projections & Economic Dimension Loadings (Biplot)\n{title_suffix}", fontsize=14, fontweight="bold")
    
    pairs = [(0, 1, axes[0]), (0, 2, axes[1])]
    for dim_x, dim_y, ax in pairs:
        for c in range(3):
            c_mask = cluster_labels[sub_idx] == c
            ax.scatter(
                X_pca[sub_idx][c_mask, dim_x],
                X_pca[sub_idx][c_mask, dim_y],
                c=CLUSTER_COLORS[c],
                alpha=0.40,
                s=20,
                label=f"Cluster {c}"
            )
            # Centroid
            ax.scatter(
                pca_centroids[c, dim_x],
                pca_centroids[c, dim_y],
                c=CLUSTER_COLORS[c],
                edgecolors="black",
                s=300,
                marker="*",
                linewidth=1.8,
                zorder=5
            )
            ax.annotate(
                f"Centroid {c}",
                (pca_centroids[c, dim_x], pca_centroids[c, dim_y]),
                xytext=(8, 8),
                textcoords="offset points",
                fontweight="bold",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8)
            )
            
        # Draw Feature Vectors (Biplot Arrows)
        scale_arrow = 3.2
        for i, name in enumerate(feat_names):
            lx = loadings[dim_x, i] * scale_arrow
            ly = loadings[dim_y, i] * scale_arrow
            if np.hypot(lx, ly) > 0.35:
                ax.arrow(0, 0, lx, ly, color="#111111", alpha=0.75, width=0.03, head_width=0.12, zorder=6)
                ax.text(lx * 1.12, ly * 1.12, name, color="#111111", fontweight="bold", fontsize=9, zorder=7)
                
        ax.set_title(f"PC{dim_x+1} vs PC{dim_y+1}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"PC{dim_x+1} ({var_exp[dim_x]*100:.1f}% Variance)", fontsize=11)
        ax.set_ylabel(f"PC{dim_y+1} ({var_exp[dim_y]*100:.1f}% Variance)", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.35)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.35)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=10, framealpha=0.85)
        
    fig2d.tight_layout()
    p2d_out = output_dir / "pca_clusters_2d.png"
    fig2d.savefig(p2d_out, dpi=200)
    plt.close(fig2d)
    print(f"Saved: {p2d_out}")
    
    return {
        "var_exp": var_exp.tolist(),
        "pca_centroids": pca_centroids.tolist(),
        "centroid_summary": centroid_summary,
        "loadings": {feat_names[i]: [float(loadings[0, i]), float(loadings[1, i]), float(loadings[2, i])] for i in range(len(feat_names))}
    }


def main():
    res3 = analyze_and_plot(
        "outputs/experiment3_voting_uniform",
        "Simulation 3: Democratic Voting + Uniform Population",
        Path("outputs/experiment3_voting_uniform/analysis")
    )
    res4 = analyze_and_plot(
        "outputs/experiment4_voting_non_uniform",
        "Simulation 4: Democratic Voting + Worker Disparity",
        Path("outputs/experiment4_voting_non_uniform/analysis")
    )
    res7 = analyze_and_plot(
        "outputs/experiment7_voting_uniform_fungible",
        "Simulation 7: Democratic Voting + Fungible Resources (+40/+5)",
        Path("outputs/experiment7_voting_uniform_fungible/analysis")
    )
    res9 = analyze_and_plot(
        "outputs/experiment9_voting_non_uniform_fungible",
        "Simulation 9: Democratic Voting + Non-Uniform + Fungible Resources (+40/+5)",
        Path("outputs/experiment9_voting_non_uniform_fungible/analysis")
    )
    
    with open("outputs/pca_comparison_summary.json", "w") as f:
        json.dump({"Sim3_Uniform": res3, "Sim4_Disparity": res4, "Sim7_Fungible": res7, "Sim9_NonUniform_Fungible": res9}, f, indent=2)
    print("\nAll PCA plots and centroid summaries successfully generated!")


if __name__ == "__main__":
    main()
