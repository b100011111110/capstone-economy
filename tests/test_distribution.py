import json
import numpy as np
import pytest

from ai_economist import foundation
from economy_rl.config import Config
from economy_rl.envs.distribution import sample_normal_efficiency, apply_environment_distribution
from economy_rl.envs.parallel import ParallelWorlds
from economy_rl.training.trainer import Trainer


def test_sample_normal_efficiency_thresholds():
    for n in [10, 50, 100]:
        z = sample_normal_efficiency(n, seed=42)
        assert len(z) == n

        n_high = np.sum(z >= 1.0)
        n_low = np.sum(z <= -1.0)

        min_required = int(np.ceil(0.20 * n))
        assert n_high >= min_required, f"For N={n}, expected >= {min_required} with z >= +1sigma, got {n_high}"
        assert n_low >= min_required, f"For N={n}, expected >= {min_required} with z <= -1sigma, got {n_low}"

        # Standard normal properties: mean within [-0.35, 0.35], std within [0.75, 1.35]
        assert -0.35 <= np.mean(z) <= 0.35
        assert 0.75 <= np.std(z) <= 1.35


def test_uniform_distribution_preserves_equal_skills():
    config = Config(distribution="uniform", n_agents=20)
    assert config.is_uniform_distribution

    env = foundation.make_env_instance(
        scenario_name="uniform/simple_wood_and_stone",
        components=[{"Gather": {}}, {"Build": {}}],
        world_size=[25, 25],
        n_agents=20,
        episode_length=50,
    )
    env.reset()
    apply_environment_distribution(env, config, seed=42)

    workers = [a for a in env.all_agents if not a.multi_action_mode]
    for w in workers:
        assert w.state["efficiency"] == 0.0
        assert w.state["effort_multiplier"] == 1.0
        assert w.state["build_skill"] == 1.0
        assert w.state["build_payment"] == 10.0
        assert w.state["bonus_gather_prob"] == 0.0


def test_non_uniform_distribution_skills_and_locations():
    config = Config(distribution="non_uniform", n_agents=30)
    assert not config.is_uniform_distribution

    env = foundation.make_env_instance(
        scenario_name="uniform/simple_wood_and_stone",
        components=[{"Gather": {}}, {"Build": {}}],
        world_size=[25, 25],
        n_agents=30,
        episode_length=50,
    )
    env.reset()
    apply_environment_distribution(env, config, seed=42)

    workers = [a for a in env.all_agents if not a.multi_action_mode]
    assert len(workers) == 30

    high_skills = [w for w in workers if w.state.get("efficiency", 0.0) >= 1.0]
    laggards = [w for w in workers if w.state.get("efficiency", 0.0) <= -1.0]

    assert len(high_skills) >= 6, f"Expected >= 6 high-skill agents (+1sigma), got {len(high_skills)}"
    assert len(laggards) >= 6, f"Expected >= 6 laggard agents (-1sigma), got {len(laggards)}"

    # Rewards remain the same for ALL agents across all tasks
    for w in workers:
        assert w.state["build_payment"] == 10.0
        assert w.state["bonus_gather_prob"] == 0.0

    # Effort scales: high skill agents require less effort, laggards require more effort
    for h in high_skills:
        assert h.state["effort_multiplier"] <= 0.65

    for l in laggards:
        assert l.state["effort_multiplier"] >= 1.35

    # Spatial proximity: high-skill agents should be significantly closer to resources than laggards
    high_dists = [h.state["initial_resource_distance"] for h in high_skills]
    laggard_dists = [l.state["initial_resource_distance"] for l in laggards]

    assert np.mean(high_dists) < np.mean(laggard_dists), (
        f"High-skill distance {np.mean(high_dists)} should be strictly less than laggards {np.mean(laggard_dists)}"
    )


def test_effort_wrapper_scales_component_labor():
    config = Config(distribution="non_uniform", n_agents=10)
    env = foundation.make_env_instance(
        scenario_name="uniform/simple_wood_and_stone",
        components=[{"Gather": {}}, {"Build": {}}],
        world_size=[25, 25],
        n_agents=10,
        episode_length=50,
    )
    env.reset()
    apply_environment_distribution(env, config, seed=42)

    workers = [a for a in env.all_agents if not a.multi_action_mode]
    w0 = workers[0]
    w1 = workers[1]
    w0.state["effort_multiplier"] = 0.5   # high skill: takes half effort
    w1.state["effort_multiplier"] = 2.0   # laggard: takes double effort

    # Test that the wrapper correctly scales labor changes from any component
    gather_comp = next(c for c in env.components if c.name == "Gather")
    orig_step = gather_comp.component_step

    # Simulate an action that would add 2.0 base labor to each agent
    def mock_step():
        w0.state["endogenous"]["Labor"] += 2.0
        w1.state["endogenous"]["Labor"] += 2.0

    # Test the wrapped behavior
    gather_comp.component_step = mock_step
    gather_comp._effort_wrapped = False
    from economy_rl.envs.distribution import _wrap_components_for_effort
    _wrap_components_for_effort(env)

    initial_labor_0 = w0.state["endogenous"]["Labor"]
    initial_labor_1 = w1.state["endogenous"]["Labor"]

    gather_comp.component_step()

    delta_0 = w0.state["endogenous"]["Labor"] - initial_labor_0
    delta_1 = w1.state["endogenous"]["Labor"] - initial_labor_1

    assert np.isclose(delta_0, 1.0)  # 2.0 * 0.5 = 1.0
    assert np.isclose(delta_1, 4.0)  # 2.0 * 2.0 = 4.0


def test_non_uniform_training_run(tmp_path):
    out_dir = tmp_path / "non_uniform_sim"
    config = Config(
        mode="normal",
        distribution="non_uniform",
        n_worlds=2,
        rollout_steps=5,
        updates=2,
        output_dir=str(out_dir),
        save_weights=False,
    )
    assert not config.is_uniform_distribution

    trainer = Trainer(config)
    trainer.train()

    summary_file = out_dir / "data" / "summary.json"
    assert summary_file.exists()
    with open(summary_file) as f:
        summary = json.load(f)
    assert summary["distribution"] == "non_uniform"

