from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from economy_rl.config import Config
from economy_rl.envs.adapter import inspect_agents
from economy_rl.envs.factory import make_environment


@dataclass
class WorldState:
    environment: Any
    observations: Dict[str, Any]
    timestep: int = 0
    episode_id: int = 0
    planner_action: Tuple[int, ...] = ()
    policy_index: int = 0


class ParallelWorlds:
    """Interleaves four independent Foundation worlds behind one vector-like API."""

    def __init__(self, config: Config):
        self.config = config
        self.worlds: List[WorldState] = []
        for world_id in range(config.n_worlds):
            environment = make_environment(config, config.seed + world_id)
            observations = environment.reset()
            self.worlds.append(WorldState(environment, observations))
        self.planner_id, self.worker_ids, self.planner_dims, self.worker_action_size = inspect_agents(
            self.worlds[0].environment
        )

    def reset_world(self, world_id: int) -> Dict[str, Any]:
        world = self.worlds[world_id]
        world.observations = world.environment.reset()
        world.timestep = 0
        world.episode_id += 1
        world.planner_action = ()
        world.policy_index = 0
        return world.observations

    def observations(self, world_id: int) -> Dict[str, Any]:
        return self.worlds[world_id].observations

    def step(self, world_id: int, actions: Dict[str, Any]):
        world = self.worlds[world_id]
        observations, raw_rewards, done, info = world.environment.step(actions)
        world.observations = observations
        world.timestep += 1
        episode_done = self._episode_done(done)
        rewards = self._apply_economic_policy(world_id, raw_rewards)
        if episode_done:
            self.reset_world(world_id)
        return world.observations, rewards, episode_done, info

    def _apply_economic_policy(self, world_id: int, raw_rewards: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tax and equal redistribution outside Foundation's fixed planner action space."""
        world = self.worlds[world_id]
        tax_rate = (0.20, 0.50, 0.80)[world.policy_index]
        worker_incomes = {
            agent_id: max(0.0, float(raw_rewards.get(agent_id, 0.0)))
            for agent_id in self.worker_ids
        }
        tax_pool = sum(worker_incomes.values()) * tax_rate
        transfer = tax_pool / max(len(self.worker_ids), 1)
        rewards = dict(raw_rewards)
        for agent_id in self.worker_ids:
            raw_value = float(raw_rewards.get(agent_id, 0.0))
            rewards[agent_id] = raw_value - tax_rate * worker_incomes[agent_id] + transfer
        worker_values = np.asarray([rewards[agent_id] for agent_id in self.worker_ids], dtype=np.float32)
        productivity = float(worker_values.sum())
        equality = 1.0 / (1.0 + float(worker_values.std()))
        rewards[self.planner_id] = productivity + 0.5 * equality
        return rewards

    @staticmethod
    def _episode_done(done: Any) -> bool:
        if isinstance(done, dict):
            return bool(done.get("__all__", False))
        return bool(done)

    def needs_planner_action(self, world_id: int) -> bool:
        return self.worlds[world_id].timestep % self.config.planner_interval == 0 or not self.worlds[world_id].planner_action

    def set_planner_action(self, world_id: int, action: Sequence[int]) -> None:
        policy_index = int(action[0])
        if policy_index not in (0, 1, 2):
            raise ValueError("planner policy must be 0, 1, or 2")
        self.worlds[world_id].policy_index = policy_index
        self.worlds[world_id].planner_action = (policy_index,)

    def planner_action(self, world_id: int) -> Tuple[int, ...]:
        return self.worlds[world_id].planner_action

    def policy_features(self, world_id: int) -> np.ndarray:
        policy_index = self.worlds[world_id].policy_index
        tax_rate = (0.20, 0.50, 0.80)[policy_index]
        return np.asarray([policy_index / 2.0, tax_rate], dtype=np.float32)

    def close(self) -> None:
        for world in self.worlds:
            close = getattr(world.environment, "close", None)
            if close:
                close()
