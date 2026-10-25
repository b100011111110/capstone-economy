import pytest
from economy_rl.config import Config
from economy_rl.envs.factory import make_environment


def test_standard_vs_fungible_build_rules():
    # 1. Standard Mode (fungible_build_resources=False)
    cfg_std = Config(
        fungible_build_resources=False,
        build_payment=40.0,
        components=["Gather", "Build"],
        n_agents=2,
        n_worlds=1,
        episode_length=50,
        seed=42,
    )
    env_std = make_environment(cfg_std, seed=42)
    env_std.reset()
    build_std = next(c for c in env_std.components if c.name == "Build")
    agent0_std = env_std.get_agent("0")

    # 2 Wood, 0 Stone cannot build in standard mode
    agent0_std.state["inventory"]["Wood"] = 2
    agent0_std.state["inventory"]["Stone"] = 0
    assert not build_std.agent_can_build(agent0_std)

    # 1 Wood, 1 Stone can build in standard mode
    agent0_std.state["inventory"]["Wood"] = 1
    agent0_std.state["inventory"]["Stone"] = 1
    assert build_std.agent_can_build(agent0_std)

    # 2. Fungible Mode (fungible_build_resources=True, cost=2)
    cfg_fungible = Config(
        fungible_build_resources=True,
        build_resource_cost=2,
        build_payment=40.0,
        components=["Gather", "Build"],
        n_agents=2,
        n_worlds=1,
        episode_length=50,
        seed=42,
    )
    env_fungible = make_environment(cfg_fungible, seed=42)
    env_fungible.reset()
    build_fungible = next(c for c in env_fungible.components if c.name == "Build")
    agent0 = env_fungible.get_agent("0")

    # Insufficient resources (1 Wood, 0 Stone)
    agent0.state["inventory"]["Wood"] = 1
    agent0.state["inventory"]["Stone"] = 0
    assert not build_fungible.agent_can_build(agent0)

    # Sufficient: 2 Wood, 0 Stone
    agent0.state["inventory"]["Wood"] = 2
    agent0.state["inventory"]["Stone"] = 0
    assert build_fungible.agent_can_build(agent0)

    # Sufficient: 0 Wood, 2 Stone
    agent0.state["inventory"]["Wood"] = 0
    agent0.state["inventory"]["Stone"] = 2
    assert build_fungible.agent_can_build(agent0)

    # Sufficient: 1 Wood, 1 Stone
    agent0.state["inventory"]["Wood"] = 1
    agent0.state["inventory"]["Stone"] = 1
    assert build_fungible.agent_can_build(agent0)

    # Test execution & resource deduction: 2 Wood, 0 Stone
    agent0.state["inventory"]["Wood"] = 2
    agent0.state["inventory"]["Stone"] = 0
    agent0.state["inventory"]["Coin"] = 0.0
    agent0.action = {"Build": 1}
    build_fungible.component_step()

    assert agent0.state["inventory"]["Wood"] == 0
    assert agent0.state["inventory"]["Stone"] == 0
    assert agent0.state["inventory"]["Coin"] == 40.0
    assert getattr(agent0, "houses_built", 0.0) == 1.0

    # Test execution with mixed resources on agent1: 1 Wood, 2 Stone
    agent1 = env_fungible.get_agent("1")
    agent1.state["inventory"]["Wood"] = 1
    agent1.state["inventory"]["Stone"] = 2
    agent1.state["inventory"]["Coin"] = 0.0
    agent1.action = {"Build": 1}
    build_fungible.component_step()

    assert agent1.state["inventory"]["Wood"] == 0
    assert agent1.state["inventory stone" if "inventory stone" in agent1.state else "inventory"]["Stone"] == 1
    assert agent1.state["inventory"]["Coin"] == 40.0
    assert getattr(agent1, "houses_built", 0.0) == 1.0

