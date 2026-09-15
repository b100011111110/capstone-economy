import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from economy_rl.config import Config
from economy_rl.envs.adapter import agent_observation, encode_actions
from economy_rl.envs.parallel import ParallelWorlds
from economy_rl.models.actor_critic import ActorCritic


class NumpyPCA:
    """Pure NumPy implementation of Principal Component Analysis."""

    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.mean: np.ndarray = None
        self.std: np.ndarray = None
        self.components_: np.ndarray = None
        self.explained_variance_ratio_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "NumpyPCA":
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-6
        X_norm = (X - self.mean) / self.std
        cov = np.cov(X_norm, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort in descending order of explained variance
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(0.0, eigenvalues[idx])
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components].T  # (n_components, n_features)
        total_var = float(np.sum(eigenvalues) + 1e-8)
        self.explained_variance_ratio_ = (eigenvalues[:self.n_components] / total_var).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean) / self.std
        return X_norm @ self.components_.T


COLOR_RED = np.array([0.89, 0.10, 0.11])     # Leader 0 (Broad Coalition / Centrist)
COLOR_GREEN = np.array([0.30, 0.69, 0.29])   # Leader 1 (Fiscal Conservative / Growth)
COLOR_BLUE = np.array([0.22, 0.49, 0.72])    # Leader 2 (Progressive Welfare)


def collect_agent_status_and_votes(
    config_path: str,
    checkpoint_path: str,
    n_episodes: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs rollouts to extract continuous 8D agent states, loyalty scores, and sampled votes."""
    config = Config.from_file(config_path)
    config.n_worlds = 2
    config.seed = seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    worlds = ParallelWorlds(config)
    planner_vectors = [worlds.get_planner_observation(w) for w in range(config.n_worlds)]
    worker_vectors = [
        np.concatenate((agent_observation(worlds.observations(0), a_id), worlds.policy_features(0)))
        for a_id in worlds.worker_ids
    ]
    planner_obs_size = max(len(v) for v in planner_vectors)
    worker_obs_size = max(len(v) for v in worker_vectors)

    leaders = [
        ActorCritic(planner_obs_size, worlds.planner_dims, config.hidden_size).to(device)
        for _ in range(config.n_leaders)
    ]
    worker = ActorCritic(worker_obs_size, (worlds.worker_action_size,), config.hidden_size).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "leaders" in ckpt:
        for idx, l_state in enumerate(ckpt["leaders"]):
            if idx < len(leaders):
                leaders[idx].load_state_dict(l_state, strict=False)
    elif "planner" in ckpt:
        for leader in leaders:
            leader.load_state_dict(ckpt["planner"], strict=False)
    if "worker" in ckpt:
        worker.load_state_dict(ckpt["worker"], strict=False)

    all_features: List[np.ndarray] = []
    all_votes: List[int] = []
    all_loyalties: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []

    for ep in range(n_episodes):
        for w_id in range(config.n_worlds):
            worlds.reset_world(w_id)
        
        for t in range(config.episode_length):
            for w_id in range(config.n_worlds):
                world_state = worlds.worlds[w_id]
                current_leader_id = worlds.current_leader_id(w_id)

                # Planner action
                p_in = torch.as_tensor(worlds.get_planner_observation(w_id), dtype=torch.float32, device=device).unsqueeze(0)
                if worlds.needs_planner_action(w_id):
                    with torch.no_grad():
                        p_act = leaders[current_leader_id].sample(p_in)[0][0].cpu().numpy()
                    worlds.set_planner_action(w_id, p_act)

                # Worker action
                w_vecs = [
                    np.concatenate((agent_observation(worlds.observations(w_id), a_id), worlds.policy_features(w_id)))
                    for a_id in worlds.worker_ids
                ]
                w_in = torch.as_tensor(np.array(w_vecs), dtype=torch.float32, device=device)
                with torch.no_grad():
                    w_act = worker.sample(w_in)[0][:, 0].cpu().numpy()

                actions = encode_actions(
                    worlds.planner_id,
                    worlds.worker_ids,
                    worlds.planner_action(w_id),
                    w_act,
                )
                _, _, done, step_info = worlds.step(w_id, actions)

                # Sample every 5 timesteps to avoid redundant near-identical frames
                if t % 5 == 0:
                    feats = world_state.clusterer.extract_agent_features(
                        agents=world_state.environment.all_agents,
                        worker_ids=worlds.worker_ids,
                        recent_rewards=np.zeros(len(worlds.worker_ids)),
                        recent_taxes=world_state.recent_taxes,
                        recent_transfers=world_state.recent_transfer,
                    )
                    loyalty_mat = world_state.loyalty_tracker.get_loyalty_matrix() # (50, 3)

                    # Softmax vote probabilities
                    temp = max(1e-4, config.voting_temperature)
                    shifted = (loyalty_mat - np.max(loyalty_mat, axis=1, keepdims=True)) / temp
                    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
                    votes = np.argmax(probs, axis=1)

                    all_features.append(feats)
                    all_votes.append(votes)
                    all_loyalties.append(loyalty_mat)
                    all_probs.append(probs)

    worlds.close()

    feat_arr = np.vstack(all_features)  # (N_samples, 8)
    vote_arr = np.concatenate(all_votes)  # (N_samples,)
    loyalty_arr = np.vstack(all_loyalties)  # (N_samples, 3)
    prob_arr = np.vstack(all_probs)  # (N_samples, 3)

    return feat_arr, vote_arr, loyalty_arr, prob_arr


def generate_pca_plots(config_path: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint paths for 3 phases
    ckpt_early = "outputs/3leaders/checkpoints/update_000010.pt"
    ckpt_mid = "outputs/3leaders/checkpoints/update_000060.pt"
    ckpt_late = "outputs/3leaders/checkpoints/update_000120.pt"

    print("Collecting agent states for Early Phase...")
    f_early, v_early, l_early, p_early = collect_agent_status_and_votes(config_path, ckpt_early, n_episodes=2, seed=11)
    
    print("Collecting agent states for Mid Phase...")
    f_mid, v_mid, l_mid, p_mid = collect_agent_status_and_votes(config_path, ckpt_mid, n_episodes=2, seed=22)

    print("Collecting agent states for Late Phase (Final Convergence)...")
    f_late, v_late, l_late, p_late = collect_agent_status_and_votes(config_path, ckpt_late, n_episodes=3, seed=33)

    # Fit PCA on global standardized features
    all_f = np.vstack([f_early, f_mid, f_late])
    pca = NumpyPCA(n_components=3)
    pca.fit(all_f)

    var_ratios = pca.explained_variance_ratio_
    print(f"PCA Explained Variance: PC1={var_ratios[0]*100:.1f}%, PC2={var_ratios[1]*100:.1f}%, PC3={var_ratios[2]*100:.1f}% (Total={np.sum(var_ratios)*100:.1f}%)")

    # Transform each phase
    pca_early = pca.transform(f_early)
    pca_mid = pca.transform(f_mid)
    pca_late = pca.transform(f_late)

    feature_names = ["Wealth", "Labor", "Wood", "Stone", "Houses", "Reward", "Tax", "Transfer"]

    # Compute RGB blended colors for smooth political alignment:
    # color_i = p_red * RED + p_green * GREEN + p_blue * BLUE
    rgb_early = p_early @ np.vstack([COLOR_RED, COLOR_GREEN, COLOR_BLUE])
    rgb_mid = p_mid @ np.vstack([COLOR_RED, COLOR_GREEN, COLOR_BLUE])
    rgb_late = p_late @ np.vstack([COLOR_RED, COLOR_GREEN, COLOR_BLUE])

    # =========================================================================
    # 1. 3D PCA SCATTER PLOT: Evolution Across Training (Early vs Mid vs Late)
    # =========================================================================
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle("3D PCA Agent Status Space & Political Voting Polarization (Red, Green, Blue)", fontsize=16, fontweight="bold", y=0.98)

    phase_data = [
        ("1. Early Phase (Update 10 / 8k Steps)", pca_early, rgb_early, v_early),
        ("2. Mid Phase (Update 60 / 48k Steps)", pca_mid, rgb_mid, v_mid),
        ("3. Late Convergence (Update 120 / 100k Steps)", pca_late, rgb_late, v_late),
    ]

    for idx, (title, p_coords, p_colors, p_votes) in enumerate(phase_data):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        
        # Subsample for clear visual density
        sub_idx = np.random.choice(len(p_coords), size=min(1200, len(p_coords)), replace=False)
        coords_sub = p_coords[sub_idx]
        colors_sub = p_colors[sub_idx]

        ax.scatter(
            coords_sub[:, 0],
            coords_sub[:, 1],
            coords_sub[:, 2],
            c=colors_sub,
            alpha=0.75,
            s=25,
            depthshade=False,
        )

        ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
        ax.set_xlabel(f"PC1 ({var_ratios[0]*100:.1f}%)", fontsize=10, labelpad=5)
        ax.set_ylabel(f"PC2 ({var_ratios[1]*100:.1f}%)", fontsize=10, labelpad=5)
        ax.set_zlabel(f"PC3 ({var_ratios[2]*100:.1f}%)", fontsize=10, labelpad=5)
        ax.view_init(elev=24, azim=135)
        ax.grid(True, alpha=0.3)

    # Custom Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='🔴 Red: Leader 0 (Broad Centrist Coalition)', markerfacecolor=COLOR_RED, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='🟢 Green: Leader 1 (Fiscal Conservative / Capital)', markerfacecolor=COLOR_GREEN, markersize=10),
        Line2D([0], [0], marker='o', color='w', label='🔵 Blue: Leader 2 (Progressive Welfare)', markerfacecolor=COLOR_BLUE, markersize=10),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_3d = output_dir / "pca_voter_clusters_3d.png"
    fig.savefig(out_3d, dpi=220)
    plt.close(fig)
    print(f"Saved: {out_3d}")

    # =========================================================================
    # 2. 2D PCA PROJECTIONS & FEATURE LOADINGS BIPLOT (Late Phase)
    # =========================================================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6.5))
    fig2.suptitle("2D PCA Projections & Economic Status Dimension Loadings (Final Converged Equilibrium)", fontsize=15, fontweight="bold")

    sub_late = np.random.choice(len(pca_late), size=min(1800, len(pca_late)), replace=False)
    late_coords = pca_late[sub_late]
    late_colors = rgb_late[sub_late]

    projections = [
        (0, 1, "PC1 vs PC2 (Primary Wealth & Labor Axes)", axes2[0]),
        (0, 2, "PC1 vs PC3 (Wealth vs Housing/Transfers)", axes2[1]),
        (1, 2, "PC2 vs PC3 (Labor Effort vs Housing)", axes2[2]),
    ]

    loadings = pca.components_  # (3, 8)

    for dim_x, dim_y, title, ax in projections:
        # Scatter points
        ax.scatter(
            late_coords[:, dim_x],
            late_coords[:, dim_y],
            c=late_colors,
            alpha=0.6,
            s=25,
        )

        # Plot PCA feature loading vectors (biplot)
        scale_factor = 3.5
        for feat_idx, feat_name in enumerate(feature_names):
            lx = loadings[dim_x, feat_idx] * scale_factor
            ly = loadings[dim_y, feat_idx] * scale_factor
            if np.hypot(lx, ly) > 0.4:
                ax.arrow(0, 0, lx, ly, color="#222222", alpha=0.75, width=0.03, head_width=0.12)
                ax.text(lx * 1.12, ly * 1.12, feat_name, color="#111111", fontweight="bold", fontsize=9)

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel(f"PC{dim_x+1} ({var_ratios[dim_x]*100:.1f}% Variance)", fontsize=10)
        ax.set_ylabel(f"PC{dim_y+1} ({var_ratios[dim_y]*100:.1f}% Variance)", fontsize=10)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3)

    fig2.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, 0.02))
    fig2.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_2d = output_dir / "pca_voter_clusters_2d.png"
    fig2.savefig(out_2d, dpi=220)
    plt.close(fig2)
    print(f"Saved: {out_2d}")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D and 2D PCA cluster plots")
    parser.add_argument("--config", default="configs/scenario_3leaders.json", help="Path to scenario config")
    parser.add_argument("--output-dir", default="outputs/3leaders/plots", help="Path to output plot directory")
    args = parser.parse_args()

    generate_pca_plots(args.config, Path(args.output_dir))


if __name__ == "__main__":
    main()
