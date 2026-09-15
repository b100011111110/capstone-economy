from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from economy_rl.config import Config
from economy_rl.envs.adapter import inspect_agents
from economy_rl.envs.factory import make_environment


TAX_RATES = (0.20, 0.50, 0.80)
REDIST_PERCENTS = (0.25, 0.50, 0.75)
HOUSE_BUILT_TAXES = (-0.10, -0.20, -0.40)
MATERIAL_SOLD_TAXES = (0.05, 0.15, 0.30)
WORK_EFFORT_TAXES = (0.00, 0.10, 0.20)
PLANNER_ACTION_DIMS = (3, 3, 3, 3, 3)


@dataclass
class WorldState:
    environment: Any
    observations: Dict[str, Any]
    timestep: int = 0
    episode_id: int = 0
    planner_action: Tuple[int, ...] = (0, 0, 0, 0, 0)
    policy_index: int = 0


class ParallelWorlds:
    """Interleaves independent Foundation worlds with a 5-head fiscal policy engine."""

    def __init__(self, config: Config):
        self.config = config
        self.worlds: List[WorldState] = []
        for world_id in range(config.n_worlds):
            environment = make_environment(config, config.seed + world_id + 1)
            observations = environment.reset()
            self.worlds.append(WorldState(environment, observations))
        self.planner_id, self.worker_ids, _, self.worker_action_size = inspect_agents(
            self.worlds[0].environment
        )
        self.planner_dims = PLANNER_ACTION_DIMS

    def reset_world(self, world_id: int) -> Dict[str, Any]:
        world = self.worlds[world_id]
        world.observations = world.environment.reset()
        world.timestep = 0
        world.episode_id += 1
        world.planner_action = (0, 0, 0, 0, 0)
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
        """Apply 5-head fiscal policy: income tax, redistribution, house subsidy, material tax, effort tax."""
        world = self.worlds[world_id]
        action = world.planner_action if len(world.planner_action) == 5 else (0, 0, 0, 0, 0)
        tax_rate = TAX_RATES[action[0]]
        redist_percent = REDIST_PERCENTS[action[1]]
        house_tax = HOUSE_BUILT_TAXES[action[2]]
        mat_tax = MATERIAL_SOLD_TAXES[action[3]]
        labor_tax = WORK_EFFORT_TAXES[action[4]]

        net_taxes: Dict[str, float] = {}
        for agent in world.environment.all_agents:
            if getattr(agent, "multi_action_mode", False):
                continue
            agent_id = str(agent.idx)
            raw_val = float(raw_rewards.get(agent_id, 0.0))
            income = max(0.0, raw_val)

            # House built subsidy (negative tax = reward boost for constructing houses)
            house_sub = -house_tax if raw_val > 5.0 else 0.0

            # Material resource gathering tax
            inv_resources = float(agent.inventory.get("Wood", 0) + agent.inventory.get("Stone", 0))
            mat_cost = mat_tax * max(0.0, inv_resources) * 0.1

            # Work effort / labor tax
            labor_val = float(agent.state.get("endogenous", {}).get("Labor", 0.0))
            labor_cost = labor_tax * max(0.0, labor_val) * 0.01

            tax_amount = (tax_rate * income) + mat_cost + labor_cost - house_sub
            net_taxes[agent_id] = tax_amount

        total_tax_collected = max(0.0, float(sum(net_taxes.values())))
        redist_pool = total_tax_collected * redist_percent
        transfer = redist_pool / max(len(self.worker_ids), 1)

        rewards = dict(raw_rewards)
        for agent_id in self.worker_ids:
            raw_val = float(raw_rewards.get(agent_id, 0.0))
            rewards[agent_id] = raw_val - net_taxes.get(agent_id, 0.0) + transfer

        worker_values = np.asarray([rewards[agent_id] for agent_id in self.worker_ids], dtype=np.float32)
        productivity = float(worker_values.sum())
        
        # Multiplicative AI Economist Social Welfare: Productivity * (1 - Gini)
        diff_sum = float(np.abs(np.subtract.outer(worker_values, worker_values)).sum())
        n_workers = len(self.worker_ids)
        positive_sum = float(np.maximum(0.0, worker_values).sum())
        denom = 2.0 * n_workers * positive_sum + 1e-8
        gini = float(np.clip(diff_sum / denom, 0.0, 1.0))
        
        treasury = total_tax_collected * (1.0 - redist_percent)
        if productivity >= 0.0:
            planner_reward = (productivity * (1.0 - gini)) + 0.05 * treasury
        else:
            planner_reward = (productivity * (1.0 + gini)) + 0.05 * treasury

        rewards[self.planner_id] = planner_reward
        return rewards

    @staticmethod
    def _episode_done(done: Any) -> bool:
        if isinstance(done, dict):
            return bool(done.get("__all__", False))
        return bool(done)

    def needs_planner_action(self, world_id: int) -> bool:
        return self.worlds[world_id].timestep % self.config.planner_interval == 0 or not self.worlds[world_id].planner_action

    def set_planner_action(self, world_id: int, action: Sequence[int]) -> None:
        action_tuple = tuple(int(a) for a in action)
        if len(action_tuple) != 5 or any(a not in (0, 1, 2) for a in action_tuple):
            raise ValueError(f"planner action must have 5 dimensions with values 0, 1, or 2, got: {action_tuple}")
        self.worlds[world_id].planner_action = action_tuple
        self.worlds[world_id].policy_index = action_tuple[0]

    def planner_action(self, world_id: int) -> Tuple[int, ...]:
        return self.worlds[world_id].planner_action

    def policy_features(self, world_id: int) -> np.ndarray:
        action = self.worlds[world_id].planner_action
        if len(action) != 5:
            action = (0, 0, 0, 0, 0)
        t_tax = TAX_RATES[action[0]]
        t_redist = REDIST_PERCENTS[action[1]]
        t_house = HOUSE_BUILT_TAXES[action[2]]
        t_mat = MATERIAL_SOLD_TAXES[action[3]]
        t_labor = WORK_EFFORT_TAXES[action[4]]
        return np.asarray([t_tax, t_redist, t_house, t_mat, t_labor], dtype=np.float32)

    def close(self) -> None:
        for world in self.worlds:
            close = getattr(world.environment, "close", None)
            if close:
                close()
