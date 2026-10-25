from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from economy_rl.config import Config
from economy_rl.envs.adapter import compute_planner_macro_features, inspect_agents
from economy_rl.envs.distribution import apply_environment_distribution
from economy_rl.envs.factory import make_environment
from economy_rl.politics.clustering import WorkerClusterer
from economy_rl.politics.elections import ElectionEngine, ElectionResult
from economy_rl.politics.loyalty import LoyaltyTracker


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
    current_leader_id: int = 0
    loyalty_tracker: Optional[LoyaltyTracker] = None
    clusterer: Optional[WorkerClusterer] = None
    cluster_assignments: Optional[np.ndarray] = None
    cluster_proportions: Optional[np.ndarray] = None
    cluster_centroids: Optional[np.ndarray] = None
    recent_taxes: Dict[str, float] = field(default_factory=dict)
    recent_transfer: float = 0.0
    last_election_result: Optional[ElectionResult] = None


class ParallelWorlds:
    """Interleaves independent Foundation worlds with a Democratic 3-Leader Engine."""

    def __init__(self, config: Config):
        self.config = config
        self.worlds: List[WorldState] = []
        for world_id in range(config.n_worlds):
            environment = make_environment(config, config.seed + world_id + 1)
            environment.reset()
            observations = apply_environment_distribution(
                environment=environment,
                config=config,
                seed=config.seed + world_id + 1,
            )
            self.worlds.append(WorldState(environment, observations))
        self.planner_id, self.worker_ids, _, self.worker_action_size = inspect_agents(
            self.worlds[0].environment
        )
        self.planner_dims = PLANNER_ACTION_DIMS
        if self.config.is_voting_mode:
            self.election_engine = ElectionEngine(
                n_leaders=config.n_leaders,
                temperature=config.voting_temperature,
                seed=config.seed,
            )

            n_workers = len(self.worker_ids)
            for world_id, world in enumerate(self.worlds):
                world.loyalty_tracker = LoyaltyTracker(
                    n_workers=n_workers,
                    n_leaders=config.n_leaders,
                    eta=config.loyalty_eta,
                    decay=config.loyalty_decay,
                    sensitivity=config.loyalty_sensitivity,
                )
                world.clusterer = WorkerClusterer(
                    n_clusters=config.n_clusters,
                    feature_dim=8,
                    seed=config.seed + world_id,
                )
                world.cluster_assignments = np.zeros(n_workers, dtype=np.int32)
                world.cluster_proportions = np.ones(config.n_clusters, dtype=np.float32) / config.n_clusters
                world.cluster_centroids = np.zeros((config.n_clusters, 8), dtype=np.float32)
        else:
            self.election_engine = None

    def reset_world(self, world_id: int) -> Dict[str, Any]:
        world = self.worlds[world_id]
        world.environment.reset()
        for a in world.environment.all_agents:
            a.houses_built = 0.0
        world.observations = apply_environment_distribution(
            environment=world.environment,
            config=self.config,
            seed=self.config.seed + (world.episode_id + 1) * 100 + world_id + 1,
        )
        world.timestep = 0
        world.episode_id += 1
        world.planner_action = (0, 0, 0, 0, 0)
        world.policy_index = 0
        world.current_leader_id = 0
        if world.loyalty_tracker:
            world.loyalty_tracker.reset()
        world.recent_taxes = {}
        world.recent_transfer = 0.0
        world.last_election_result = None
        return world.observations

    def observations(self, world_id: int) -> Dict[str, Any]:
        return self.worlds[world_id].observations

    def step(self, world_id: int, actions: Dict[str, Any]):
        world = self.worlds[world_id]
        pickup_rew = float(getattr(self.config, "resource_pickup_reward", 0.0) or 0.0)
        pre_inv = None
        if pickup_rew > 0.0:
            pre_inv = {
                str(a.idx): (float(a.inventory.get("Wood", 0.0)), float(a.inventory.get("Stone", 0.0)))
                for a in world.environment.all_agents
                if not getattr(a, "multi_action_mode", False)
            }

        observations, raw_rewards, done, info = world.environment.step(actions)
        if pre_inv is not None and pickup_rew > 0.0:
            raw_rewards = dict(raw_rewards)
            for a in world.environment.all_agents:
                if getattr(a, "multi_action_mode", False):
                    continue
                a_id = str(a.idx)
                prev_w, prev_s = pre_inv.get(a_id, (0.0, 0.0))
                curr_w = float(a.inventory.get("Wood", 0.0))
                curr_s = float(a.inventory.get("Stone", 0.0))
                delta_gathered = max(0.0, curr_w - prev_w) + max(0.0, curr_s - prev_s)
                if delta_gathered > 0:
                    raw_rewards[a_id] = float(raw_rewards.get(a_id, 0.0)) + pickup_rew * delta_gathered

        world.observations = observations
        world.timestep += 1
        episode_done = self._episode_done(done)
        rewards, step_info = self._apply_economic_policy(world_id, raw_rewards)
        merged_info = dict(info) if isinstance(info, dict) else {}
        merged_info.update(step_info)
        if episode_done:
            self.reset_world(world_id)
        return world.observations, rewards, episode_done, merged_info

    def _apply_economic_policy(self, world_id: int, raw_rewards: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply 5-head fiscal policy, update dynamic clusters & loyalty, and run elections."""
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

        world.recent_taxes = net_taxes
        total_tax_collected = max(0.0, float(sum(net_taxes.values())))
        redist_pool = total_tax_collected * redist_percent
        transfer = redist_pool / max(len(self.worker_ids), 1)
        world.recent_transfer = transfer

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
            social_welfare = (productivity * (1.0 - gini)) + 0.05 * treasury
        else:
            social_welfare = (productivity * (1.0 + gini)) + 0.05 * treasury

        # Extract per-agent snapshot (resources, effort, loyalties)
        workers = [a for a in world.environment.all_agents if not getattr(a, "multi_action_mode", False)]
        loyalty_matrix = world.loyalty_tracker.get_loyalty_matrix() if (self.config.is_voting_mode and world.loyalty_tracker is not None) else None

        agent_records = []
        for i, agent in enumerate(workers):
            a_id = str(agent.idx)
            inv = getattr(agent, "inventory", {})
            coin = float(inv.get("Coin", 0.0))
            wood = float(inv.get("Wood", 0.0))
            stone = float(inv.get("Stone", 0.0))

            endogenous = getattr(agent, "state", {}).get("endogenous", {})
            labor = float(endogenous.get("Labor", 0.0))
            houses = float(getattr(agent, "houses_built", 0.0) or endogenous.get("Houses", 0.0))

            effort_mult = float(agent.state.get("effort_multiplier", 1.0))
            efficiency = float(agent.state.get("efficiency", 0.0))

            if loyalty_matrix is not None and i < len(loyalty_matrix):
                agent_loyalties = [float(v) for v in loyalty_matrix[i]]
                pref_vote = int(np.argmax(agent_loyalties))
            else:
                agent_loyalties = [0.0] * self.config.n_leaders
                pref_vote = 0

            agent_records.append({
                "agent_id": a_id,
                "coin": coin,
                "wood": wood,
                "stone": stone,
                "houses": houses,
                "labor": labor,
                "effort_multiplier": effort_mult,
                "efficiency": efficiency,
                "loyalty_leader_0": agent_loyalties[0] if len(agent_loyalties) > 0 else 0.0,
                "loyalty_leader_1": agent_loyalties[1] if len(agent_loyalties) > 1 else 0.0,
                "loyalty_leader_2": agent_loyalties[2] if len(agent_loyalties) > 2 else 0.0,
                "loyalties": json.dumps(agent_loyalties),
                "preferred_vote": pref_vote,
            })

        if not self.config.is_voting_mode:
            rewards[self.planner_id] = float(social_welfare)
            step_info = {
                "social_welfare": float(social_welfare),
                "productivity": productivity,
                "gini": gini,
                "treasury": treasury,
                "current_leader_id": 0,
                "election_held": False,
                "election_result": None,
                "agent_records": agent_records,
            }
            return rewards, step_info

        # 1. Update Dynamic Loyalty Tracker [-1.0, 1.0]
        delta_u = world.loyalty_tracker.update(
            incumbent_id=world.current_leader_id,
            worker_rewards=worker_values,
        )
        mean_loyalties = world.loyalty_tracker.get_mean_leader_loyalty()

        # 2. Extract N-Dimensional Status Features & Adaptively Cluster Workers
        features = world.clusterer.extract_agent_features(
            agents=world.environment.all_agents,
            worker_ids=self.worker_ids,
            recent_rewards=worker_values,
            recent_taxes=net_taxes,
            recent_transfers=transfer,
        )
        assignments, proportions, centroids = world.clusterer.fit_predict(features)
        world.cluster_assignments = assignments
        world.cluster_proportions = proportions
        world.cluster_centroids = centroids

        # 3. Check for Democratic Election
        election_held = False
        election_result = None
        if world.timestep > 0 and world.timestep % self.config.election_interval == 0:
            election_result = self.election_engine.run_election(
                loyalty_matrix=world.loyalty_tracker.get_loyalty_matrix(),
                incumbent_id=world.current_leader_id,
                cluster_assignments=world.cluster_assignments,
                n_clusters=self.config.n_clusters,
            )
            world.last_election_result = election_result
            world.current_leader_id = election_result.winner_id
            election_held = True

        # 4. Multi-Leader Reward Distribution
        # The incumbent leader who governed receives social welfare + voter approval + vote share bonus
        incumbent_id = world.current_leader_id
        approval_bonus = self.config.approval_reward_coeff * float(mean_loyalties[incumbent_id])
        vote_bonus = 0.0
        if world.last_election_result:
            vote_bonus = self.config.vote_reward_coeff * float(world.last_election_result.vote_shares[incumbent_id])

        leader_rewards = np.zeros(self.config.n_leaders, dtype=np.float32)
        leader_rewards[incumbent_id] = social_welfare + approval_bonus + vote_bonus

        rewards[self.planner_id] = float(leader_rewards[incumbent_id])

        step_info = {
            "social_welfare": social_welfare,
            "productivity": productivity,
            "gini": gini,
            "treasury": treasury,
            "current_leader_id": world.current_leader_id,
            "mean_leader_loyalty": mean_loyalties.tolist(),
            "leader_rewards": leader_rewards.tolist(),
            "cluster_proportions": proportions.tolist(),
            "election_held": election_held,
            "election_result": election_result,
            "agent_records": agent_records,
        }
        return rewards, step_info

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

    def current_leader_id(self, world_id: int) -> int:
        if not self.config.is_voting_mode:
            return 0
        return self.worlds[world_id].current_leader_id

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

    def get_planner_observation(self, world_id: int) -> np.ndarray:
        world = self.worlds[world_id]
        pol_features = self.policy_features(world_id)
        if not self.config.is_voting_mode:
            return compute_planner_macro_features(
                environment=world.environment,
                policy_features=pol_features,
                timestep=world.timestep,
                episode_length=self.config.episode_length,
                include_politics=False,
            )
        cluster_loyalties = world.loyalty_tracker.get_cluster_loyalty(
            cluster_assignments=world.cluster_assignments,
            n_clusters=self.config.n_clusters,
        )
        mean_loyalties = world.loyalty_tracker.get_mean_leader_loyalty()
        return compute_planner_macro_features(
            environment=world.environment,
            policy_features=pol_features,
            timestep=world.timestep,
            episode_length=self.config.episode_length,
            cluster_proportions=world.cluster_proportions,
            cluster_loyalty_matrix=cluster_loyalties,
            mean_leader_loyalty=mean_loyalties,
            current_leader_id=world.current_leader_id,
            n_leaders=self.config.n_leaders,
            n_clusters=self.config.n_clusters,
            include_politics=True,
        )

    def close(self) -> None:
        for world in self.worlds:
            close = getattr(world.environment, "close", None)
            if close:
                close()
