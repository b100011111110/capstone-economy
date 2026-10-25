import pytest
from economy_rl.config import Config
from economy_rl.envs.factory import make_environment
from economy_rl.envs.parallel import ParallelWorlds


def test_build_payment_and_pickup_reward_config():
    cfg = Config(
        build_payment=40.0,
        resource_pickup_reward=5.0,
        components=["Gather", "Build"],
        n_agents=4,
        n_worlds=1,
        episode_length=50,
        seed=1,
    )
    env = make_environment(cfg, seed=1)
    build_comp = next((c for c in env.components if c.name == "Build"), None)
    assert build_comp is not None
    assert build_comp.payment == 40.0

    # Test ParallelWorlds step with pickup reward
    worlds = ParallelWorlds(cfg)
    assert worlds.config.build_payment == 40.0
    assert worlds.config.resource_pickup_reward == 5.0
    obs = worlds.reset_world(0)
    assert obs is not None
    worlds.close()

