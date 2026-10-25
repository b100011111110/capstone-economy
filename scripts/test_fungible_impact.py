"""Fast empirical verification of Fungible Resources vs Standard Rules on trained weights."""

import json
from economy_rl.config import Config
from economy_rl.training.trainer import Trainer


def run_eval(fungible: bool, n_steps: int = 300):
    mode_name = "FUNGIBLE (Wood+Stone >= 2)" if fungible else "STANDARD (1 Wood AND 1 Stone)"
    print(f"\n{'='*70}\nRunning Evaluation: {mode_name}\n{'='*70}")

    config = Config(
        scenario_name="uniform/simple_wood_and_stone",
        mode="voting",
        distribution="uniform",
        components=["Gather", "Build"],
        world_size=[25, 25],
        n_agents=50,
        n_worlds=4,
        episode_length=300,
        planner_interval=25,
        election_interval=100,
        n_leaders=3,
        n_clusters=3,
        build_payment=40.0,
        resource_pickup_reward=5.0,
        fungible_build_resources=fungible,
        build_resource_cost=2,
        checkpoint_path="outputs/experiment6_voting_uniform_shaped/checkpoints/update_000050.pt",
        output_dir=f"outputs/eval_fungible_{fungible}",
        rollout_steps=100,
        updates=3,
        seed=42,
    )

    trainer = Trainer(config)
    print("Collecting 3 rollouts (300 steps per world across 4 parallel worlds)...")
    trainer.train()

    # Read the results from data/agent_episodes.csv
    import csv
    with open(f"{config.output_dir}/data/agent_episodes.csv") as f:
        rows = list(csv.DictReader(f))

    total_coins = sum(float(r["coin"]) for r in rows)
    total_houses = sum(float(r["houses"]) for r in rows)
    total_labor = sum(float(r["labor"]) for r in rows)
    total_wood = sum(float(r["wood"]) for r in rows)
    total_stone = sum(float(r["stone"]) for r in rows)

    print(f"Results for {mode_name}:")
    print(f"  Episodes Finished: {len(set(r['episode'] for r in rows))}")
    print(f"  Houses Built:      {total_houses:.0f}")
    print(f"  Coins Generated:   {total_coins:.0f}")
    print(f"  Remaining Wood:    {total_wood:.0f}")
    print(f"  Remaining Stone:   {total_stone:.0f}")
    print(f"  Labor Expended:    {total_labor:.0f}")

    return {
        "fungible": fungible,
        "houses": total_houses,
        "coins": total_coins,
        "wood": total_wood,
        "stone": total_stone,
    }


if __name__ == "__main__":
    res_fungible = run_eval(fungible=True)
    print("\nVerification complete.")

