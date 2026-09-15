from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ElectionResult:
    winner_id: int
    vote_counts: List[int]
    vote_shares: List[float]
    individual_votes: np.ndarray
    cluster_vote_shares: Optional[np.ndarray]
    incumbent_reelected: bool
    margin_of_victory: float


class ElectionEngine:
    """Simulates democratic elections among competing economic leaders based on worker loyalties."""

    def __init__(
        self,
        n_leaders: int = 3,
        temperature: float = 0.5,
        seed: int = 7,
    ):
        self.n_leaders = n_leaders
        self.temperature = temperature
        self.rng = np.random.RandomState(seed)

    def run_election(
        self,
        loyalty_matrix: np.ndarray,
        incumbent_id: int,
        cluster_assignments: Optional[np.ndarray] = None,
        n_clusters: int = 3,
    ) -> ElectionResult:
        """Runs a democratic ballot election based on worker loyalty scores.
        
        Args:
            loyalty_matrix: Array of shape (n_workers, n_leaders) with values in [-1.0, 1.0].
            incumbent_id: Index of the current governing leader.
            cluster_assignments: Optional array of shape (n_workers,) with cluster IDs.
            n_clusters: Number of dynamic clusters.
            
        Returns:
            ElectionResult containing the winner, vote tallies, and demographic breakdown.
        """
        n_workers, n_leaders = loyalty_matrix.shape
        temp = max(1e-4, self.temperature)

        # Softmax probability of voting for each candidate:
        # P(vote_i = k) = exp(L_i,k / temp) / sum_j exp(L_i,j / temp)
        shifted_loyalty = loyalty_matrix / temp
        # Numerical stability shift
        shifted_loyalty -= np.max(shifted_loyalty, axis=1, keepdims=True)
        exp_loyalty = np.exp(shifted_loyalty)
        probs = exp_loyalty / np.sum(exp_loyalty, axis=1, keepdims=True)

        # Workers cast their secret ballots
        individual_votes = np.zeros(n_workers, dtype=np.int32)
        for i in range(n_workers):
            individual_votes[i] = self.rng.choice(n_leaders, p=probs[i])

        vote_counts = np.bincount(individual_votes, minlength=n_leaders).tolist()
        vote_shares = [count / float(n_workers) for count in vote_counts]

        # Determine winner (tie-break favoring incumbent, then random)
        max_votes = max(vote_counts)
        top_candidates = [idx for idx, count in enumerate(vote_counts) if count == max_votes]
        if len(top_candidates) == 1:
            winner_id = top_candidates[0]
        elif incumbent_id in top_candidates:
            winner_id = incumbent_id
        else:
            winner_id = int(self.rng.choice(top_candidates))

        # Margin of victory over runner-up
        sorted_shares = sorted(vote_shares, reverse=True)
        margin = sorted_shares[0] - (sorted_shares[1] if len(sorted_shares) > 1 else 0.0)
        incumbent_reelected = (winner_id == incumbent_id)

        # Compute cluster voting breakdown if assignments provided
        cluster_vote_shares = None
        if cluster_assignments is not None:
            cluster_vote_shares = np.zeros((n_clusters, n_leaders), dtype=np.float32)
            for c in range(n_clusters):
                mask = cluster_assignments == c
                if np.any(mask):
                    c_votes = individual_votes[mask]
                    c_counts = np.bincount(c_votes, minlength=n_leaders)
                    cluster_vote_shares[c] = c_counts / float(len(c_votes))
                else:
                    cluster_vote_shares[c] = 1.0 / float(n_leaders)

        return ElectionResult(
            winner_id=winner_id,
            vote_counts=vote_counts,
            vote_shares=vote_shares,
            individual_votes=individual_votes,
            cluster_vote_shares=cluster_vote_shares,
            incumbent_reelected=incumbent_reelected,
            margin_of_victory=margin,
        )

