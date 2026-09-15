from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-critic with one categorical head per discrete action dimension."""

    def __init__(self, observation_size: int, action_sizes: Sequence[int], hidden_size: int = 128):
        super().__init__()
        self.action_sizes = tuple(int(size) for size in action_sizes)
        self.encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_heads = nn.ModuleList(nn.Linear(hidden_size, size) for size in self.action_sizes)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        features = self.encoder(observations)
        logits = [head(features) for head in self.actor_heads]
        values = self.value_head(features).squeeze(-1)
        return logits, values

    def sample(self, observations: torch.Tensor, deterministic: bool = False):
        logits, values = self(observations)
        actions = []
        log_probabilities = []
        entropies = []
        for action_logits in logits:
            distribution = Categorical(logits=action_logits)
            action = torch.argmax(action_logits, dim=-1) if deterministic else distribution.sample()
            actions.append(action)
            log_probabilities.append(distribution.log_prob(action))
            entropies.append(distribution.entropy())
        action_tensor = torch.stack(actions, dim=-1)
        log_probability = torch.stack(log_probabilities, dim=-1).sum(dim=-1)
        entropy = torch.stack(entropies, dim=-1).sum(dim=-1)
        return action_tensor, log_probability, values, entropy

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        logits, values = self(observations)
        log_probabilities = []
        entropies = []
        for index, action_logits in enumerate(logits):
            distribution = Categorical(logits=action_logits)
            log_probabilities.append(distribution.log_prob(actions[:, index]))
            entropies.append(distribution.entropy())
        return (
            torch.stack(log_probabilities, dim=-1).sum(dim=-1),
            values,
            torch.stack(entropies, dim=-1).sum(dim=-1),
        )
