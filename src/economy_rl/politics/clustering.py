from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


class WorkerClusterer:
    """Dynamic, data-driven N-dimensional clustering of heterogeneous economic agents.
    
    Extracts continuous multi-dimensional status features (wealth, labor, resources, 
    production, transfers, taxes) and clusters workers adaptively without fixed 
    socioeconomic preconceptions.
    """

    def __init__(self, n_clusters: int = 3, feature_dim: int = 8, seed: int = 7):
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        self.centroids: np.ndarray = np.zeros((n_clusters, feature_dim), dtype=np.float32)
        self.fitted = False

    def extract_agent_features(
        self,
        agents: Sequence[Any],
        worker_ids: Sequence[str],
        recent_rewards: np.ndarray,
        recent_taxes: Dict[str, float],
        recent_transfers: float,
    ) -> np.ndarray:
        """Extracts N-dimensional continuous status vectors for each worker.
        
        Dimensions:
            0: Wealth (Coin inventory)
            1: Labor effort (Endogenous state)
            2: Wood inventory
            3: Stone inventory
            4: Houses built
            5: Current step reward / utility
            6: Net tax paid
            7: Net transfer received
        """
        features: List[List[float]] = []
        for i, agent_id in enumerate(worker_ids):
            # Find agent object
            agent = next((a for a in agents if str(a.idx) == agent_id), None)
            if agent is None:
                features.append([0.0] * self.feature_dim)
                continue

            inv = getattr(agent, "inventory", {})
            coin = float(inv.get("Coin", 0.0))
            wood = float(inv.get("Wood", 0.0))
            stone = float(inv.get("Stone", 0.0))

            endogenous = getattr(agent, "state", {}).get("endogenous", {})
            labor = float(endogenous.get("Labor", 0.0))
            houses = float(getattr(agent, "houses_built", 0.0) or endogenous.get("Houses", 0.0))

            step_reward = float(recent_rewards[i]) if i < len(recent_rewards) else 0.0
            tax_paid = float(recent_taxes.get(agent_id, 0.0))
            transfer_rec = float(recent_transfers)

            vec = [
                coin,
                labor,
                wood,
                stone,
                houses,
                step_reward,
                tax_paid,
                transfer_rec,
            ]
            features.append(vec)

        return np.asarray(features, dtype=np.float32)

    def fit_predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Clusters workers dynamically based on their N-dimensional status vectors.
        
        Args:
            features: Array of shape (n_workers, feature_dim).
            
        Returns:
            cluster_assignments: Array of shape (n_workers,) with cluster IDs in 0..K-1.
            cluster_proportions: Array of shape (n_clusters,) with population shares.
            cluster_centroids: Array of shape (n_clusters, feature_dim) in standardized space.
        """
        n_samples = len(features)
        if n_samples == 0:
            return (
                np.zeros(0, dtype=np.int32),
                np.ones(self.n_clusters, dtype=np.float32) / self.n_clusters,
                np.zeros((self.n_clusters, self.feature_dim), dtype=np.float32),
            )

        # Standardize features online: (X - mu) / (sigma + eps)
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) + 1e-6
        norm_features = (features - mean) / std

        # Initialize centroids if not fitted or dimension mismatch
        if not self.fitted or self.centroids.shape != (self.n_clusters, self.feature_dim):
            # K-means++ initialization or random selection
            indices = self.rng.choice(n_samples, size=min(self.n_clusters, n_samples), replace=False)
            self.centroids = norm_features[indices].copy()
            if len(self.centroids) < self.n_clusters:
                pad = np.zeros((self.n_clusters - len(self.centroids), self.feature_dim), dtype=np.float32)
                self.centroids = np.vstack([self.centroids, pad])
            self.fitted = True

        # Run adaptive K-means iterations (5-10 iterations for online tracking)
        assignments = np.zeros(n_samples, dtype=np.int32)
        for _ in range(8):
            # Compute Euclidean distances: (N, K)
            dists = np.linalg.norm(norm_features[:, None, :] - self.centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)

            # Update centroids with smoothing
            for k in range(self.n_clusters):
                mask = assignments == k
                if np.any(mask):
                    new_center = np.mean(norm_features[mask], axis=0)
                    self.centroids[k] = 0.8 * self.centroids[k] + 0.2 * new_center

        # Compute cluster proportions
        proportions = np.zeros(self.n_clusters, dtype=np.float32)
        for k in range(self.n_clusters):
            proportions[k] = np.sum(assignments == k) / float(n_samples)

        return assignments, proportions, self.centroids.copy()

