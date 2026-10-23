import json
import shutil
from pathlib import Path
import pytest
import torch

from economy_rl.config import Config
from economy_rl.envs.parallel import ParallelWorlds
from economy_rl.training.trainer import Trainer


def test_config_mode_and_weight_flags(tmp_path):
    # Test normal mode via mode="normal"
    c_norm = Config(mode="normal")
    assert not c_norm.is_voting_mode
    assert c_norm.n_leaders == 1

    # Test voting mode via mode="voting"
    c_vote = Config(mode="voting", n_leaders=3)
    assert c_vote.is_voting_mode
    assert c_vote.n_leaders == 3

    # Test alias voting=False
    c_alias_norm = Config(voting=False)
    assert not c_alias_norm.is_voting_mode
    assert c_alias_norm.mode == "normal"
    assert c_alias_norm.n_leaders == 1

    # Test alias enable_voting=True
    c_alias_vote = Config(enable_voting=True)
    assert c_alias_vote.is_voting_mode
    assert c_alias_vote.mode == "voting"

    # Test weight flags and aliases
    c_weights = Config(
        save_checkpoints=False,
        save_interval=5,
        max_checkpoints=2,
        delete_old_checkpoints=True,
        delete_checkpoints_after_training=True,
    )
    assert c_weights.save_weights is False
    assert c_weights.checkpoint_interval == 5
    assert c_weights.keep_last_n_checkpoints == 2
    assert c_weights.delete_old_checkpoints is True
    assert c_weights.delete_checkpoints_after_training is True

    # Test loading from json file with aliases and extra fields
    json_path = tmp_path / "test_config.json"
    data = {
        "mode": "normal",
        "save_weights": True,
        "checkpoint_interval": 3,
        "keep_last_n_checkpoints": 2,
        "unknown_extra_field": "should_be_ignored",
    }
    with open(json_path, "w") as f:
        json.dump(data, f)

    loaded = Config.from_file(str(json_path))
    assert not loaded.is_voting_mode
    assert loaded.mode == "normal"
    assert loaded.save_weights is True
    assert loaded.checkpoint_interval == 3
    assert loaded.keep_last_n_checkpoints == 2


def test_normal_simulation_run(tmp_path):
    out_dir = tmp_path / "normal_run"
    config = Config(
        mode="normal",
        n_worlds=2,
        rollout_steps=5,
        updates=2,
        save_weights=True,
        checkpoint_interval=1,
        output_dir=str(out_dir),
    )
    assert not config.is_voting_mode

    trainer = Trainer(config)
    assert len(trainer.leaders) == 1
    assert trainer.worlds.election_engine is None

    trainer.train()

    # In normal mode, elections and loyalty csvs should NOT be generated
    assert not (out_dir / "data" / "elections.csv").exists()
    assert not (out_dir / "data" / "loyalty.csv").exists()

    # Steps and summary should exist
    assert (out_dir / "data" / "steps.csv").exists()
    assert (out_dir / "data" / "summary.json").exists()

    with open(out_dir / "data" / "summary.json") as f:
        summary = json.load(f)
    assert summary["mode"] == "normal"
    assert "n_leaders" not in summary

    # Checkpoint was saved
    ckpts = list((out_dir / "checkpoints").glob("*.pt"))
    assert len(ckpts) > 0
    loaded = torch.load(ckpts[0], map_location="cpu")
    assert loaded["mode"] == "normal"
    assert "planner" in loaded


def test_weight_retention_and_deletion(tmp_path):
    # Test keep_last_n_checkpoints = 2
    out_dir = tmp_path / "retention_run"
    config = Config(
        mode="normal",
        n_worlds=2,
        rollout_steps=5,
        updates=4,
        save_weights=True,
        checkpoint_interval=1,
        keep_last_n_checkpoints=2,
        save_final_weights=False,
        output_dir=str(out_dir),
    )
    trainer = Trainer(config)
    trainer.train()

    ckpt_dir = out_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("update_*.pt"))
    # Out of 4 updates (update 0, 1, 2, 3), only 2 should remain (update_000002, update_000003)
    assert len(ckpts) == 2
    assert ckpts[0].name == "update_000002.pt"
    assert ckpts[1].name == "update_000003.pt"


def test_no_weight_saving(tmp_path):
    out_dir = tmp_path / "no_save_run"
    config = Config(
        mode="normal",
        n_worlds=2,
        rollout_steps=5,
        updates=2,
        save_weights=False,
        output_dir=str(out_dir),
    )
    trainer = Trainer(config)
    trainer.train()

    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        assert len(list(ckpt_dir.glob("*.pt"))) == 0


def test_delete_checkpoints_after_training(tmp_path):
    out_dir = tmp_path / "del_after_run"
    config = Config(
        mode="normal",
        n_worlds=2,
        rollout_steps=5,
        updates=2,
        save_weights=True,
        checkpoint_interval=1,
        delete_checkpoints_after_training=True,
        output_dir=str(out_dir),
    )
    trainer = Trainer(config)
    trainer.train()

    ckpt_dir = out_dir / "checkpoints"
    if ckpt_dir.exists():
        assert len(list(ckpt_dir.glob("*.pt"))) == 0

