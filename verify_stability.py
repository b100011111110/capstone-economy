import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def verify_normal():
    base = Path("outputs/stability_normal")
    assert base.exists(), "stability_normal output directory missing"

    # 1. Check data files
    steps = pd.read_csv(base / "data" / "steps.csv")
    assert len(steps) == 360, f"Expected 360 steps, got {len(steps)}"
    assert not steps.isna().any().any(), "NaN found in steps.csv"

    updates = pd.read_csv(base / "data" / "updates.csv")
    assert len(updates) == 6, f"Expected 6 updates, got {len(updates)}"
    assert not updates.isna().any().any(), "NaN found in updates.csv"

    # In normal mode, no elections or loyalty files should exist
    assert not (base / "data" / "elections.csv").exists(), "elections.csv should not exist in normal mode"
    assert not (base / "data" / "loyalty.csv").exists(), "loyalty.csv should not exist in normal mode"

    # 2. Check summary
    with open(base / "data" / "summary.json") as f:
        summary = json.load(f)
    assert summary["mode"] == "normal"
    assert summary["save_weights"] is True
    assert summary["keep_last_n_checkpoints"] == 2

    # 3. Check checkpoints
    ckpt_dir = base / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("update_*.pt"))
    assert len(ckpts) == 2, f"Expected 2 periodic checkpoints, got {len(ckpts)}"
    assert ckpts[0].name == "update_000002.pt"
    assert ckpts[1].name == "update_000004.pt"
    assert (ckpt_dir / "final.pt").exists(), "final.pt missing"

    # Verify checkpoint contents
    final_ckpt = torch.load(ckpt_dir / "final.pt", map_location="cpu")
    assert final_ckpt["mode"] == "normal"
    assert "planner" in final_ckpt
    assert "worker" in final_ckpt

    # 4. Check analysis
    assert (base / "analysis" / "economy_training.png").exists()
    assert (base / "analysis" / "convergence.json").exists()
    print("✓ Normal simulation stability verification PASSED")

def verify_voting():
    base = Path("outputs/stability_voting")
    assert base.exists(), "stability_voting output directory missing"

    # 1. Check data files
    steps = pd.read_csv(base / "data" / "steps.csv")
    assert len(steps) == 360, f"Expected 360 steps, got {len(steps)}"
    assert not steps.isna().any().any(), "NaN found in steps.csv"

    updates = pd.read_csv(base / "data" / "updates.csv")
    assert len(updates) == 6, f"Expected 6 updates, got {len(updates)}"
    assert not updates.isna().any().any(), "NaN found in updates.csv"

    # In voting mode, elections and loyalty files MUST exist
    assert (base / "data" / "elections.csv").exists(), "elections.csv missing in voting mode"
    assert (base / "data" / "loyalty.csv").exists(), "loyalty.csv missing in voting mode"

    elections = pd.read_csv(base / "data" / "elections.csv")
    assert len(elections) > 0, "No elections recorded"
    for _, row in elections.iterrows():
        shares = json.loads(row["vote_shares"])
        assert len(shares) == 3, "Expected 3 candidate vote shares"
        assert np.isclose(sum(shares), 1.0, atol=1e-3), "Vote shares must sum to 1.0"
        assert 0 <= row["winner_id"] < 3, "Winner ID out of range"

    loyalty = pd.read_csv(base / "data" / "loyalty.csv")
    assert len(loyalty) > 0, "No loyalty recorded"
    for col in ["leader_0_loyalty", "leader_1_loyalty", "leader_2_loyalty"]:
        assert (loyalty[col] >= -1.0 - 1e-4).all() and (loyalty[col] <= 1.0 + 1e-4).all(), f"Loyalty {col} out of [-1, 1] range"

    # 2. Check summary
    with open(base / "data" / "summary.json") as f:
        summary = json.load(f)
    assert summary["mode"] == "voting"
    assert summary["n_leaders"] == 3
    assert summary["election_interval"] == 20
    assert summary["save_weights"] is True
    assert summary["keep_last_n_checkpoints"] == 2

    # 3. Check checkpoints
    ckpt_dir = base / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("update_*.pt"))
    assert len(ckpts) == 2, f"Expected 2 periodic checkpoints, got {len(ckpts)}"
    assert ckpts[0].name == "update_000002.pt"
    assert ckpts[1].name == "update_000004.pt"
    assert (ckpt_dir / "final.pt").exists(), "final.pt missing"

    # Verify checkpoint contents
    final_ckpt = torch.load(ckpt_dir / "final.pt", map_location="cpu")
    assert final_ckpt["mode"] == "voting"
    assert "leaders" in final_ckpt
    assert len(final_ckpt["leaders"]) == 3
    assert "worker" in final_ckpt

    # 4. Check analysis
    assert (base / "analysis" / "economy_training.png").exists()
    assert (base / "analysis" / "convergence.json").exists()
    print("✓ Voting simulation stability verification PASSED")

def verify_non_uniform():
    base = Path("outputs/sample_non_uniform")
    assert base.exists(), "sample_non_uniform output directory missing"

    # 1. Check data files
    steps = pd.read_csv(base / "data" / "steps.csv")
    assert len(steps) >= 40, f"Expected at least 40 steps, got {len(steps)}"
    assert not steps.isna().any().any(), "NaN found in steps.csv"

    updates = pd.read_csv(base / "data" / "updates.csv")
    assert len(updates) >= 2, f"Expected at least 2 updates, got {len(updates)}"
    assert not updates.isna().any().any(), "NaN found in updates.csv"

    # Check per-agent data logging
    agent_steps = pd.read_csv(base / "data" / "agent_steps.csv")
    assert not agent_steps.empty, "agent_steps.csv is empty"
    required_agent_cols = [
        "agent_id", "action", "step_reward", "cumulative_reward",
        "coin", "wood", "stone", "houses",
        "labor", "step_labor", "effort_multiplier", "efficiency",
        "loyalty_leader_0", "loyalty_leader_1", "loyalty_leader_2",
        "loyalties", "preferred_vote",
    ]
    for col in required_agent_cols:
        assert col in agent_steps.columns, f"Missing {col} in agent_steps.csv"
    assert not agent_steps[required_agent_cols].isna().any().any(), "NaN found in agent_steps.csv"

    # 2. Check summary
    with open(base / "data" / "summary.json") as f:
        summary = json.load(f)
    assert summary["mode"] == "voting"
    assert summary["distribution"] == "non_uniform"
    assert summary["save_weights"] is True
    assert summary["keep_last_n_checkpoints"] == 1

    # 3. Check checkpoints (retention = 1)
    ckpt_dir = base / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("update_*.pt"))
    assert len(ckpts) == 1, f"Expected 1 periodic checkpoint, got {len(ckpts)}"
    assert ckpts[0].name == "update_000001.pt"

    print("✓ Non-uniform simulation stability verification PASSED")

if __name__ == "__main__":
    verify_normal()
    verify_voting()
    verify_non_uniform()
    print("\nALL STABILITY VERIFICATIONS COMPLETED SUCCESSFULLY!")

