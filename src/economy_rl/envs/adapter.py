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
