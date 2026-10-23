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

    # Simulation Mode: "voting" (or "democratic") vs "normal" (or "standard")
    mode: str = "voting"
    simulation_mode: str = None
    voting: bool = None
    enable_voting: bool = None

    # Distribution Mode: "uniform" vs "non_uniform" (or "heterogeneous")
    distribution: str = "uniform"
    distribution_type: str = None
    agent_distribution: str = None
    resource_distribution: str = None

    # Checkpoint & Weight Management Parameters
    save_weights: bool = True
    save_checkpoints: bool = None
    checkpoint_interval: int = 10
    save_interval: int = None
    save_final_weights: bool = True
    save_final_checkpoint: bool = None
    keep_last_n_checkpoints: int = None
    max_checkpoints: int = None
    delete_old_checkpoints: bool = False
    delete_checkpoints_after_training: bool = False

    # Per-Agent Metrics & History Logging
    log_agent_metrics: bool = True
    save_agent_metrics: bool = None
    capture_agent_data: bool = None
    agent_log_interval: int = 1

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

    @property
    def is_voting_mode(self) -> bool:
        if self.enable_voting is not None:
            return bool(self.enable_voting)
        if self.voting is not None:
            return bool(self.voting)
        if self.simulation_mode is not None:
            return str(self.simulation_mode).strip().lower() in ("voting", "democratic", "politics", "multi_leader", "true", "1")
        if self.mode is not None:
            return str(self.mode).strip().lower() in ("voting", "democratic", "politics", "multi_leader", "true", "1")
        return self.n_leaders > 1

    @property
    def is_uniform_distribution(self) -> bool:
        val = self.distribution or self.distribution_type or self.agent_distribution or self.resource_distribution or "uniform"
        return str(val).strip().lower() in ("uniform", "homogeneous", "standard", "equal")

    def __post_init__(self):
        # Resolve mode aliases
        if self.simulation_mode is not None:
            self.mode = str(self.simulation_mode)
        elif self.voting is not None:
            self.mode = "voting" if self.voting else "normal"
        elif self.enable_voting is not None:
            self.mode = "voting" if self.enable_voting else "normal"

        if not self.is_voting_mode:
            self.mode = "normal"
            self.n_leaders = 1
        else:
            self.mode = "voting"

        # Resolve distribution aliases
        if self.distribution_type is not None:
            self.distribution = str(self.distribution_type)
        elif self.agent_distribution is not None:
            self.distribution = str(self.agent_distribution)
        elif self.resource_distribution is not None:
            self.distribution = str(self.resource_distribution)

        if not self.is_uniform_distribution:
            self.distribution = "non_uniform"
        else:
            self.distribution = "uniform"

        # Resolve weight and checkpoint aliases
        if self.save_checkpoints is not None:
            self.save_weights = bool(self.save_checkpoints)
        if self.save_interval is not None:
            self.checkpoint_interval = int(self.save_interval)
        if self.save_final_checkpoint is not None:
            self.save_final_weights = bool(self.save_final_checkpoint)
        if self.max_checkpoints is not None:
            self.keep_last_n_checkpoints = int(self.max_checkpoints)

        # Resolve agent metric logging aliases
        if self.save_agent_metrics is not None:
            self.log_agent_metrics = bool(self.save_agent_metrics)
        elif self.capture_agent_data is not None:
            self.log_agent_metrics = bool(self.capture_agent_data)
        if self.agent_log_interval <= 0:
            self.agent_log_interval = 1

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
            raw_data = json.load(handle)
        import dataclasses
        known_field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in raw_data.items() if k in known_field_names}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
