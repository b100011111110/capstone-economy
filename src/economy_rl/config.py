from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Config:
    scenario_name: str = "uniform/simple_wood_and_stone"
    components: List[str] = None
    world_size: List[int] = None
    n_agents: int = 50
    n_worlds: int = 4
    episode_length: int = 300
    planner_interval: int = 30
    rollout_steps: int = 120
    updates: int = 1000
    learning_rate: float = 0.005
    planner_learning_rate: float = 0.0005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    worker_entropy_coefficient: float = 0.05
    planner_entropy_coefficient: float = 0.01
    normalize_rewards: bool = True
    ppo_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    seed: int = 7
    device: str = "auto"
    output_dir: str = "outputs"
    checkpoint_path: str = None
    # Political Economy & Dynamic Clustering Parameters
    n_leaders: int = 3
    election_interval: int = 100
    n_clusters: int = 3
    loyalty_eta: float = 0.15
    loyalty_decay: float = 0.02
    loyalty_sensitivity: float = 1.0
    voting_temperature: float = 0.5
    approval_reward_coeff: float = 0.5
    vote_reward_coeff: float = 0.5

    def __post_init__(self):
        if self.components is None:
            self.components = ["Gather", "Build"]
        if self.world_size is None:
            self.world_size = [25, 25]
        if self.world_size != [25, 25]:
            raise ValueError("world_size must be [25, 25]")
        if self.n_agents <= 0 or self.n_worlds <= 0:
            raise ValueError("n_agents and n_worlds must be positive")
        if self.planner_interval <= 0:
            raise ValueError("planner_interval must be positive")
        if self.planner_learning_rate is None:
            self.planner_learning_rate = self.learning_rate / 10.0

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
