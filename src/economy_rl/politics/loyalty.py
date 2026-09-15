from typing import Optional
import numpy as np


class LoyaltyTracker:
    """Tracks dynamic voter loyalty in [-1.0, 1.0] towards competing political leaders.
    
    Attributes:
        -1.0: Can't stand (detests leader / negative life impact)
         0.0: Indifferent (neutral baseline)
        +1.0: Loves leader (strong approval / positive life improvement)
    """

    def __init__(
        self,
        n_workers: int = 50,
        n_leaders: int = 3,
        eta: float = 0.15,
        decay: float = 0.02,
        sensitivity: float = 1.0,
        baseline_alpha: float = 0.05,
    ):
        self.n_workers = n_workers
        self.n_leaders = n_leaders
        self.eta = eta
        self.decay = decay
        self.sensitivity = sensitivity
        self.baseline_alpha = baseline_alpha

        # Strict initialization: all voters start completely indifferent (0.0)
        self.loyalty_matrix = np.zeros((n_workers, n_leaders), dtype=np.float32)
        self.baseline_rewards = np.zeros(n_workers, dtype=np.float32)
        self.has_initial_baseline = False

    def reset(self) -> None:
        """Reset loyalty matrix and baselines."""
        self.loyalty_matrix.fill(0.0)
        self.baseline_rewards.fill(0.0)
        self.has_initial_baseline = False

    def update(
        self,
        incumbent_id: int,
        worker_rewards: np.ndarray,
    ) -> np.ndarray:
        """Update worker loyalties based on experienced reward delta under incumbent.
        
        Args:
            incumbent_id: Index of the currently governing leader (0 to n_leaders - 1).
            worker_rewards: Array of shape (n_workers,) containing current step post-tax rewards.
            
        Returns:
            delta_utilities: Array of shape (n_workers,) representing life improvement deltas.
        """
        rewards = np.asarray(worker_rewards, dtype=np.float32)
        if not self.has_initial_baseline:
            self.baseline_rewards = np.copy(rewards)
            self.has_initial_baseline = True

        # Delta utility: how much life improved relative to rolling expectation
        delta_u = rewards - self.baseline_rewards

        # 1. Update loyalty for the incumbent leader
        # delta_u > 0 => loyalty increases towards +1.0
        # delta_u < 0 => loyalty decreases towards -1.0
        incumbent_shift = self.eta * np.tanh(self.sensitivity * delta_u)
        self.loyalty_matrix[:, incumbent_id] = np.clip(
            self.loyalty_matrix[:, incumbent_id] + incumbent_shift,
            -1.0,
            1.0,
        )

        # 2. Memory decay for out-of-office leaders (gradually fade towards indifference 0.0)
        for leader_id in range(self.n_leaders):
            if leader_id != incumbent_id:
                self.loyalty_matrix[:, leader_id] = (
                    1.0 - self.decay
                ) * self.loyalty_matrix[:, leader_id]

        # 3. Update rolling baseline expected utility
        self.baseline_rewards = (
            (1.0 - self.baseline_alpha) * self.baseline_rewards
            + self.baseline_alpha * rewards
        )

        return delta_u

    def get_loyalty_matrix(self) -> np.ndarray:
        """Returns the full (n_workers, n_leaders) loyalty matrix."""
        return np.copy(self.loyalty_matrix)

    def get_mean_leader_loyalty(self) -> np.ndarray:
        """Returns average approval rating per leader across all workers (shape: n_leaders)."""
        return np.mean(self.loyalty_matrix, axis=0)

    def get_cluster_loyalty(
        self,
        cluster_assignments: np.ndarray,
        n_clusters: int = 3,
    ) -> np.ndarray:
        """Computes mean loyalty to each leader broken down by dynamic cluster.
        
        Returns:
            matrix of shape (n_clusters, n_leaders) with mean loyalty in [-1.0, 1.0].
        """
        cluster_loyalty = np.zeros((n_clusters, self.n_leaders), dtype=np.float32)
        for c in range(n_clusters):
            mask = cluster_assignments == c
            if np.any(mask):
                cluster_loyalty[c] = np.mean(self.loyalty_matrix[mask], axis=0)
            else:
                cluster_loyalty[c] = 0.0
        return cluster_loyalty

