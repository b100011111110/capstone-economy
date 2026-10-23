import json
from pathlib import Path
import pandas as pd
import pytest

from economy_rl.config import Config
from economy_rl.training.trainer import Trainer


def test_per_agent_metrics_captured(tmp_path):
    out_dir = tmp_path / "test_agent_capture"
    config = Config(
        mode="voting",
        distribution="non_uniform",
        n_agents=10,
        n_worlds=2,
        episode_length=10,
        rollout_steps=10,
        updates=1,
        save_weights=False,
        log_agent_metrics=True,
        agent_log_interval=1,
        output_dir=str(out_dir),
    )

    trainer = Trainer(config)
    trainer.train()

    data_dir = out_dir / "data"
    steps_csv = data_dir / "agent_steps.csv"
    episodes_csv = data_dir / "agent_episodes.csv"

    assert steps_csv.exists(), "agent_steps.csv was not created"
    assert episodes_csv.exists(), "agent_episodes.csv was not created"

    df_steps = pd.read_csv(steps_csv)
    assert not df_steps.empty, "agent_steps.csv is empty"

    # Verify that all 10 agents are present
    agent_ids = set(df_steps["agent_id"].astype(str))
    expected_ids = {str(i) for i in range(10)}
    assert expected_ids.issubset(agent_ids), f"Missing agents in agent_steps: {expected_ids - agent_ids}"

    # Verify required fields are present
    required_fields = [
        "global_step", "world", "episode", "timestep", "agent_id",
        # Actions taken
        "action",
        # Rewards accumulated
        "step_reward", "cumulative_reward",
        # Resources accumulated
        "coin", "wood", "stone", "houses",
        # Efforts done
        "labor", "step_labor", "effort_multiplier", "efficiency",
        # Loyalties
        "loyalty_leader_0", "loyalty_leader_1", "loyalty_leader_2",
        "loyalties", "preferred_vote",
    ]
    for col in required_fields:
        assert col in df_steps.columns, f"Column '{col}' missing from agent_steps.csv"

    # Verify data validity: no NaN values
    assert not df_steps[required_fields].isna().any().any(), "NaN values detected in agent_steps.csv"

    # Verify effort metrics
    assert (df_steps["labor"] >= 0.0).all(), "Labor should be non-negative"
    assert (df_steps["step_labor"] >= 0.0).all(), "Step labor should be non-negative"
    assert (df_steps["effort_multiplier"] > 0.0).all(), "Effort multiplier should be positive"

    # Verify loyalties are within [-1.0, 1.0]
    for col in ["loyalty_leader_0", "loyalty_leader_1", "loyalty_leader_2"]:
        assert (df_steps[col] >= -1.0 - 1e-4).all() and (df_steps[col] <= 1.0 + 1e-4).all(), f"Loyalty {col} out of range"

    # Verify JSON loyalties list parses correctly
    sample_loyalties = json.loads(df_steps["loyalties"].iloc[0])
    assert len(sample_loyalties) == 3

    # Check episode level summary
    df_episodes = pd.read_csv(episodes_csv)
    assert not df_episodes.empty, "agent_episodes.csv is empty"
    assert len(df_episodes) >= 10, f"Expected at least 10 agent episode rows, got {len(df_episodes)}"
    for col in required_fields:
        assert col in df_episodes.columns, f"Column '{col}' missing from agent_episodes.csv"


def test_agent_metrics_can_be_disabled(tmp_path):
    out_dir = tmp_path / "test_agent_disabled"
    config = Config(
        mode="normal",
        distribution="uniform",
        n_agents=10,
        n_worlds=1,
        rollout_steps=5,
        updates=1,
        save_weights=False,
        log_agent_metrics=False,
        output_dir=str(out_dir),
    )

    trainer = Trainer(config)
    trainer.train()

    data_dir = out_dir / "data"
    assert not (data_dir / "agent_steps.csv").exists(), "agent_steps.csv should not exist when disabled"
    assert not (data_dir / "agent_episodes.csv").exists(), "agent_episodes.csv should not exist when disabled"

