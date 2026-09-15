import numpy as np
import pytest

from economy_rl.politics.loyalty import LoyaltyTracker
from economy_rl.politics.clustering import WorkerClusterer
from economy_rl.politics.elections import ElectionEngine


def test_loyalty_initialization():
    tracker = LoyaltyTracker(n_workers=10, n_leaders=3)
    # Check that all voter loyalties start strictly at 0.0 (indifferent)
    assert np.all(tracker.get_loyalty_matrix() == 0.0)
    assert np.all(tracker.get_mean_leader_loyalty() == 0.0)


def test_loyalty_bounds_and_dynamics():
    tracker = LoyaltyTracker(n_workers=4, n_leaders=3, eta=0.2, decay=0.05, sensitivity=1.0)
    
    # First step: set baseline rewards = [1.0, 2.0, 3.0, 4.0]
    tracker.update(incumbent_id=0, worker_rewards=np.array([1.0, 2.0, 3.0, 4.0]))
    
    # Second step: Leader 0 improves rewards significantly for workers 0 and 1, hurts worker 2 and 3
    new_rewards = np.array([5.0, 6.0, 0.0, 0.0])
    tracker.update(incumbent_id=0, worker_rewards=new_rewards)
    
    mat = tracker.get_loyalty_matrix()
    # Workers 0 and 1 should love leader 0 (> 0.0)
    assert mat[0, 0] > 0.0
    assert mat[1, 0] > 0.0
    # Workers 2 and 3 should dislike leader 0 (< 0.0)
    assert mat[2, 0] < 0.0
    assert mat[3, 0] < 0.0
    
    # Leaders 1 and 2 were out of office, their loyalty should remain 0.0 (or decayed towards 0)
    assert np.all(mat[:, 1] == 0.0)
    assert np.all(mat[:, 2] == 0.0)
    
    # Push loyalty with extreme values for many steps to verify hard clipping [-1.0, 1.0]
    for _ in range(50):
        tracker.update(incumbent_id=0, worker_rewards=np.array([100.0, 100.0, -100.0, -100.0]))
    
    mat_extreme = tracker.get_loyalty_matrix()
    assert np.all(mat_extreme <= 1.0)
    assert np.all(mat_extreme >= -1.0)
    assert np.isclose(mat_extreme[0, 0], 1.0)
    assert np.isclose(mat_extreme[2, 0], -1.0)


def test_worker_clustering():
    clusterer = WorkerClusterer(n_clusters=3, feature_dim=4, seed=42)
    # Generate 3 distinct mock feature groups
    g1 = np.random.randn(10, 4) + np.array([10.0, 0.0, 0.0, 0.0])
    g2 = np.random.randn(10, 4) + np.array([0.0, 10.0, 0.0, 0.0])
    g3 = np.random.randn(10, 4) + np.array([0.0, 0.0, 10.0, 0.0])
    features = np.vstack([g1, g2, g3])
    
    assignments, proportions, centroids = clusterer.fit_predict(features)
    assert len(assignments) == 30
    assert len(proportions) == 3
    assert np.isclose(np.sum(proportions), 1.0)
    assert centroids.shape == (3, 4)
    # Check that all 3 clusters are populated
    assert len(np.unique(assignments)) == 3


def test_election_engine():
    engine = ElectionEngine(n_leaders=3, temperature=0.1, seed=42)
    
    # Create loyalty matrix where leader 1 is strongly loved by all workers
    loyalty = np.zeros((20, 3), dtype=np.float32)
    loyalty[:, 1] = 0.9  # Loves Leader 1
    loyalty[:, 0] = -0.5
    loyalty[:, 2] = -0.5
    
    result = engine.run_election(loyalty, incumbent_id=0)
    assert result.winner_id == 1
    assert result.vote_shares[1] > 0.8
    assert not result.incumbent_reelected

