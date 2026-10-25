#!/usr/bin/env python3
"""Simulation Status Monitor.
Reads active or latest experiment progress and displays real-time metrics.
"""

import json
import sys
import time
from pathlib import Path


def get_active_dir() -> Path:
    candidates = [
        Path("outputs/experiment14_normal_non_uniform_anti_crash"),
        Path("outputs/experiment13_normal_uniform_anti_crash"),
        Path("outputs/experiment12_voting_uniform_anti_crash"),
        Path("outputs/experiment11_voting_anti_crash"),
        Path("outputs/experiment9_voting_non_uniform_fungible"),
        Path("outputs/experiment10_normal_non_uniform_fungible"),
        Path("outputs/experiment7_voting_uniform_fungible"),
        Path("outputs/experiment8_normal_uniform_fungible"),
        Path("outputs/experiment6_voting_uniform_shaped"),
        Path("outputs/experiment5_normal_uniform_shaped"),
        Path("outputs/experiment4_voting_non_uniform"),
        Path("outputs/experiment3_voting_uniform"),
        Path("outputs/experiment2_normal_non_uniform"),
        Path("outputs/experiment1_normal_uniform"),
    ]
    for c in candidates:
        if (c / "history.jsonl").exists():
            return c
    return candidates[0]


def main():
    exp_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else get_active_dir()
    history_path = exp_dir / "history.jsonl"
    config_path = Path("configs") / f"{exp_dir.name}.json"

    print("=" * 65)
    print(f"Monitoring: {exp_dir}")
    print("=" * 65)

    cfg = {}
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception:
            pass

    target_updates = cfg.get("updates", 64)
    rollout_steps = cfg.get("rollout_steps", 100)
    n_worlds = cfg.get("n_worlds", 16)
    target_steps = target_updates * rollout_steps * n_worlds
    planner_interval = cfg.get("planner_interval", 25)
    target_intervals = target_steps // planner_interval
    election_interval = cfg.get("election_interval", 100)
    is_voting = cfg.get("mode") == "voting"
    target_elections = (target_steps // election_interval) if is_voting else 0

    if not history_path.exists():
        print(f"No history.jsonl found in {exp_dir} (simulation starting or not initialized)")
        print(f"Target Configuration: {target_updates} updates, {target_steps} steps, {target_elections} elections")
        return

    updates = []
    planner_intervals = []

    with open(history_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("type")
            if t == "update":
                updates.append(rec)
            elif t == "planner_interval":
                planner_intervals.append(rec)

    n_u = len(updates)
    n_p = len(planner_intervals)
    curr_steps = updates[-1]["global_step"] if updates else 0
    pct = (n_u / target_updates) * 100 if target_updates > 0 else 0

    print(f"Status: {'COMPLETED' if n_u >= target_updates else 'IN PROGRESS'}")
    print(f"Updates:          {n_u:03d} / {target_updates:03d} ({pct:.1f}%)")
    print(f"Policy Changes:   {n_p:05d} / {target_intervals:05d} ({(n_p / max(1, target_intervals)) * 100:.1f}%)")
    print(f"Global Steps:     {curr_steps:06d} / {target_steps:06d} ({(curr_steps / max(1, target_steps)) * 100:.1f}%)")

    # Election stats if voting mode
    elections_csv = exp_dir / "data" / "elections.csv"
    if is_voting and elections_csv.exists():
        import csv
        elections = []
        with open(elections_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                elections.append(row)
        n_elec = len(elections)
        elec_pct = (n_elec / max(1, target_elections)) * 100
        print(f"Elections Held:   {n_elec:05d} / {target_elections:05d} ({elec_pct:.1f}%)")
        if elections:
            recent_elec = elections[-100:]
            reelected = sum(1 for e in recent_elec if e.get("incumbent_reelected", "").lower() in ("true", "1"))
            re_rate = (reelected / len(recent_elec)) * 100
            print(f"  Incumbent Reelection Rate (last {len(recent_elec)}): {re_rate:.1f}%")

    if updates:
        latest = updates[-1]
        w = latest.get("worker", {})
        p = latest.get("planner", {})
        print("-" * 65)
        print(f"Latest Update {latest.get('update')} (Global Step {curr_steps}):")
        if "steps_per_second" in latest:
            print(f"  Throughput:        {latest.get('steps_per_second', 0.0):.1f} steps/s ({latest.get('elapsed_seconds', 0.0):.2f}s per update)")
        print(f"  Worker Entropy:    {w.get('entropy', 0.0):.4f} nats")
        print(f"  Worker Total Loss: {w.get('loss', 0.0):.4f}")
        print(f"  Planner Entropy:   {p.get('entropy', 0.0):.4f} nats")
        print(f"  Planner Loss:      {p.get('loss', 0.0):.4f}")

    if planner_intervals:
        recent = planner_intervals[-100:]
        recent_rewards = [r["reward"] for r in recent]
        avg_rew = sum(recent_rewards) / len(recent_rewards)
        print("-" * 65)
        print(f"Recent Mean Planner Reward (last {len(recent)} intervals): {avg_rew:.2f}")

    print("=" * 65)


if __name__ == "__main__":
    main()
