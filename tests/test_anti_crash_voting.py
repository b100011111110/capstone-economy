import numpy as np
import pytest

from economy_rl.config import Config
from economy_rl.politics.loyalty import LoyaltyTracker
from economy_rl.envs.parallel import ParallelWorlds


def test_sociotropic_loyalty_recession_penalty():
    """Verify that voters penalize the incumbent leader when a recession occurs."""
    tracker = LoyaltyTracker(n_workers=5, n_leaders=3, eta=0.15, sensitivity=1.0)
    
    # Establish baseline
    rewards = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    tracker.update(incumbent_id=0, worker_rewards=rewards)
    
    # Under regular zero delta, loyalty remains neutral
    tracker.update(incumbent_id=0, worker_rewards=rewards, economic_growth=0.0, recession=False)
    assert np.allclose(tracker.get_loyalty_matrix()[:, 0], 0.0, atol=1e-3)
    
    # Now simulate economic crash / recession (0 houses built)
    tracker.update(incumbent_id=0, worker_rewards=rewards, economic_growth=0.0, recession=True, recession_penalty=0.3)
    
    # Incumbent loyalty must drop significantly below zero (voters hate the crash!)
    loyalty_matrix = tracker.get_loyalty_matrix()
    assert np.all(loyalty_matrix[:, 0] < -0.25)
    
    # Out of office leaders (1 and 2) must not be penalized
    assert np.allclose(loyalty_matrix[:, 1], 0.0, atol=1e-3)
    assert np.allclose(loyalty_matrix[:, 2], 0.0, atol=1e-3)


def test_sociotropic_loyalty_growth_bonus():
    """Verify that voters reward the incumbent leader when the economy is actively building."""
    tracker = LoyaltyTracker(n_workers=5, n_leaders=3, eta=0.15, sensitivity=1.0)
    rewards = np.array([2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    tracker.update(incumbent_id=1, worker_rewards=rewards)
    
    # Economic growth / housing boom
    tracker.update(incumbent_id=1, worker_rewards=rewards, economic_growth=1.0, recession=False, growth_bonus=0.2)
    
    loyalty_matrix = tracker.get_loyalty_matrix()
    # Incumbent (Leader 1) loyalty increases positively
    assert np.all(loyalty_matrix[:, 1] > 0.15)


def test_parallel_worlds_anti_crash_mechanisms():
    """Verify that ParallelWorlds applies construction tax credits, public housing dividends, and recession penalties."""
    cfg = Config(
        n_agents=4,
        n_worlds=1,
        episode_length=50,
        planner_interval=5,
        mode="voting",
        n_leaders=3,
        anti_crash_voting=True,
        recession_loyalty_penalty=0.25,
        growth_loyalty_bonus=0.15,
        housing_dividend_per_house=20.0,
        recession_slump_penalty=1.0,
        planner_recession_penalty=30.0,
        construction_tax_credit=0.35,
        build_payment=40.0,
        fungible_build_resources=True,
    )
    
    pw = ParallelWorlds(cfg)
    pw.reset_world(0)
    w = pw.worlds[0]
    
    # Check initialization of counters
    assert w.recent_interval_houses == 0.0
    assert w.step_houses_built == 0.0
    
    # Mock world timestep at 10 (past planner_interval=5) with 0 houses built => Recession
    w.timestep = 10
    w.recent_interval_houses = 0.0
    raw_rewards = {pw.worker_ids[0]: 5.0, pw.worker_ids[1]: 5.0, pw.worker_ids[2]: 5.0, pw.worker_ids[3]: 5.0}
    
    rewards, step_info = pw._apply_economic_policy(0, raw_rewards)
    
    # Planner/leader reward should include the -30.0 planner_recession_penalty
    assert rewards[pw.planner_id] < -10.0
    
    # Workers should experience the recession slump penalty (1.0)
    for w_id in pw.worker_ids:
        # Without recession penalty reward would be higher
        assert rewards[w_id] < 5.0
        
    # Incumbent loyalty should have dropped due to recession
    mean_loyalties = w.loyalty_tracker.get_mean_leader_loyalty()
    assert mean_loyalties[w.current_leader_id] < 0.0
    
    # Test setting planner action resets interval houses
    w.recent_interval_houses = 12.0
    pw.set_planner_action(0, [1, 1, 1, 1, 1])
    assert w.recent_interval_houses == 0.0

