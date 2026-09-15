import argparse
import json
from pathlib import Path

import numpy as np
import torch

from economy_rl.config import Config
from economy_rl.envs.adapter import encode_actions
from economy_rl.envs.parallel import ParallelWorlds
from economy_rl.envs.adapter import agent_observation
from economy_rl.models.actor_critic import ActorCritic


POLICIES = {
    0: "tax_20_percent",
    1: "tax_50_percent",
    2: "tax_80_percent",
}


def run_demo(output_path: str, checkpoint: str = "") -> None:
    config = Config()
    worlds = ParallelWorlds(config)
    records = []
    try:
        worker_ids = worlds.worker_ids
        worker_vectors = []
        for agent_id in worker_ids:
            worker_vectors.append(np.concatenate((
                agent_observation(worlds.observations(0), agent_id),
                worlds.policy_features(0),
            )))
        observation_size = max(len(vector) for vector in worker_vectors)
        policy = ActorCritic(observation_size, (worlds.worker_action_size,), config.hidden_size)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            policy.load_state_dict(state["worker"])
        policy.eval()
        for policy_index in (0, 1, 2):
            local_worlds = ParallelWorlds(config)
            try:
                local_worlds.set_planner_action(0, [policy_index])
                vectors = [np.concatenate((agent_observation(local_worlds.observations(0), agent_id), local_worlds.policy_features(0))) for agent_id in worker_ids]
                padded = np.zeros((len(vectors), observation_size), dtype=np.float32)
                for index, vector in enumerate(vectors):
                    padded[index, :len(vector)] = vector
                with torch.no_grad():
                    worker_actions = policy.sample(torch.as_tensor(padded), deterministic=True)[0][:, 0].numpy()
                action = encode_actions(local_worlds.planner_id, worker_ids, [policy_index], worker_actions)
                _, rewards, _, _ = local_worlds.step(0, action)
                worker_rewards = np.asarray([float(rewards[agent_id]) for agent_id in worker_ids])
            finally:
                local_worlds.close()
            records.append({
                "policy_index": policy_index,
                "policy_name": POLICIES[policy_index],
                "tax_rate": [0.20, 0.50, 0.80][policy_index],
                "worker_mean_reward": float(worker_rewards.mean()),
                "worker_total_reward": float(worker_rewards.sum()),
                "worker_reward_std": float(worker_rewards.std()),
                "planner_reward": float(rewards[worlds.planner_id]),
                "worker_action_mean": float(worker_actions.mean()),
                "worker_action_mode": int(np.bincount(worker_actions).argmax()),
            })
    finally:
        worlds.close()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/policy_reaction_demo.json")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    run_demo(args.output, args.checkpoint)
