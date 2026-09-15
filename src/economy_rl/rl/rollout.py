from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    log_probability: float
    value: float
    reward: float
    done: bool
    next_value: float


@dataclass
class RolloutBuffer:
    transitions: List[Transition] = field(default_factory=list)

    def add(self, **kwargs: Any) -> None:
        self.transitions.append(Transition(**kwargs))

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        advantages = np.zeros(len(self.transitions), dtype=np.float32)
        returns = np.zeros(len(self.transitions), dtype=np.float32)
        running_advantage = 0.0
        for index in reversed(range(len(self.transitions))):
            transition = self.transitions[index]
            next_value = 0.0 if transition.done else transition.next_value
            delta = transition.reward + gamma * next_value - transition.value
            running_advantage = delta + gamma * gae_lambda * (0.0 if transition.done else running_advantage)
            advantages[index] = running_advantage
            returns[index] = advantages[index] + transition.value
        return returns, advantages

    def as_arrays(self):
        return {
            "observations": np.asarray([item.observation for item in self.transitions], dtype=np.float32),
            "actions": np.asarray([item.action for item in self.transitions], dtype=np.int64),
            "log_probabilities": np.asarray([item.log_probability for item in self.transitions], dtype=np.float32),
            "values": np.asarray([item.value for item in self.transitions], dtype=np.float32),
            "rewards": np.asarray([item.reward for item in self.transitions], dtype=np.float32),
            "dones": np.asarray([item.done for item in self.transitions], dtype=np.float32),
        }
