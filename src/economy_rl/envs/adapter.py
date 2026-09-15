from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


def _flatten(value: Any) -> List[float]:
    if isinstance(value, dict):
        values: List[float] = []
        for key in sorted(value, key=str):
            values.extend(_flatten(value[key]))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(_flatten(item))
        return values
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False).reshape(-1).tolist()
    if isinstance(value, (bool, int, float, np.number)):
        return [float(value)]
    return []


def flatten_observation(observation: Any) -> np.ndarray:
    """Convert an AI Economist observation into a stable one-dimensional vector."""
    return np.asarray(_flatten(observation), dtype=np.float32)


def agent_observation(observations: Dict[str, Any], agent_id: str) -> np.ndarray:
    value = observations.get(agent_id, observations.get(str(agent_id), {}))
    return flatten_observation(value)


def inspect_agents(environment: Any) -> Tuple[str, List[str], Tuple[int, ...], int]:
    """Return planner id, worker ids, planner action dimensions, worker action size."""
    planner_id = "p"
    worker_ids: List[str] = []
    planner_dims: Tuple[int, ...] = ()
    worker_action_size = 0
    for agent in environment.all_agents:
        agent_id = str(agent.idx)
        action_spaces = agent.action_spaces
        if agent.multi_action_mode:
            planner_id = agent_id
            planner_dims = tuple(int(size) for size in action_spaces)
        else:
            worker_ids.append(agent_id)
            worker_action_size = max(worker_action_size, int(action_spaces))
    if not planner_dims:
        raise ValueError("No multi-action planner was found in the environment")
    if not worker_ids or not worker_action_size:
        raise ValueError("No worker action space was found in the environment")
    return planner_id, worker_ids, planner_dims, worker_action_size


def encode_actions(
    planner_id: str,
    worker_ids: Sequence[str],
    planner_action: Sequence[int],
    worker_actions: Sequence[int],
) -> Dict[str, Any]:
    # Foundation's built-in planner action is fixed in this scenario; the economic
    # planner policy is applied by ParallelWorlds before rewards are returned.
    actions: Dict[str, Any] = {planner_id: [0]}
    actions.update({agent_id: int(action) for agent_id, action in zip(worker_ids, worker_actions)})
    return actions


def compute_planner_macro_features(
    environment: Any,
    policy_features: np.ndarray,
    timestep: int,
    episode_length: int = 300,
    cluster_proportions: Sequence[float] = None,
    cluster_loyalty_matrix: np.ndarray = None,
    mean_leader_loyalty: Sequence[float] = None,
    current_leader_id: int = 0,
    n_leaders: int = 3,
    n_clusters: int = 3,
) -> np.ndarray:
    """Extract 4 Gaussian parametric curves + Gini + Dynamic Clustering + Voter Loyalty."""
    worker_agents = [a for a in environment.all_agents if not a.multi_action_mode]
    coins = np.array([float(a.inventory.get("Coin", 0.0)) for a in worker_agents], dtype=np.float32)
    wood = np.array([float(a.inventory.get("Wood", 0.0)) for a in worker_agents], dtype=np.float32)
    stone = np.array([float(a.inventory.get("Stone", 0.0)) for a in worker_agents], dtype=np.float32)
    labor = np.array([float(a.state.get("endogenous", {}).get("Labor", 0.0)) for a in worker_agents], dtype=np.float32)

    total_wealth = coins + 2.0 * wood + 2.0 * stone
    sorted_wealth = np.sort(total_wealth)

    # 4 Gaussian Curves (Quartile distribution: Poorest, Lower-Middle, Upper-Middle, Top 25%)
    n = len(sorted_wealth)
    q_chunks = np.array_split(sorted_wealth, 4)
    gaussian_features: List[float] = []
    for q in q_chunks:
        pi_k = float(len(q)) / float(max(n, 1))
        mu_k = float(np.mean(q)) / 50.0
        sigma_k = float(np.std(q)) / 20.0
        gaussian_features.extend([pi_k, mu_k, sigma_k])

    # Gini Coefficient & Poverty Rate
    diff_sum = float(np.abs(np.subtract.outer(total_wealth, total_wealth)).sum())
    denom = float(2.0 * n * total_wealth.sum() + 1e-8)
    gini = float(diff_sum / denom)
    poverty_rate = float(np.mean(total_wealth == 0))

    # Macroeconomic Aggregates & Progress
    mean_coins = float(np.mean(coins)) / 50.0
    mean_wood = float(np.mean(wood)) / 20.0
    mean_stone = float(np.mean(stone)) / 20.0
    mean_labor = float(np.mean(labor)) / 100.0
    progress = float(timestep) / float(max(episode_length, 1))

    # Political Economy Features:
    # 1. Cluster population shares
    if cluster_proportions is None:
        cluster_props = [1.0 / n_clusters] * n_clusters
    else:
        cluster_props = list(cluster_proportions)

    # 2. Cluster loyalty matrix (n_clusters x n_leaders)
    if cluster_loyalty_matrix is None:
        cluster_loyalties = [0.0] * (n_clusters * n_leaders)
    else:
        cluster_loyalties = cluster_loyalty_matrix.flatten().tolist()

    # 3. Overall leader approval ratings (n_leaders)
    if mean_leader_loyalty is None:
        leader_approvals = [0.0] * n_leaders
    else:
        leader_approvals = list(mean_leader_loyalty)

    # 4. Incumbent one-hot indicator
    incumbent_one_hot = [1.0 if i == current_leader_id else 0.0 for i in range(n_leaders)]

    macro_features = (
        gaussian_features
        + [gini, poverty_rate, mean_coins, mean_wood, mean_stone, mean_labor, progress]
        + list(policy_features)
        + cluster_props
        + cluster_loyalties
        + leader_approvals
        + incumbent_one_hot
    )
    return np.asarray(macro_features, dtype=np.float32)
