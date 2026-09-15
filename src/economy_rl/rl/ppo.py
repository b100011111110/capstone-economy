from typing import Dict

import numpy as np
import torch
from torch import nn

from economy_rl.models.actor_critic import ActorCritic
from economy_rl.rl.rollout import RolloutBuffer


def update_policy(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    if not buffer.transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    arrays = buffer.as_arrays()
    returns, advantages = buffer.compute_returns_and_advantages(gamma, gae_lambda)
    observations = torch.as_tensor(arrays["observations"], device=device)
    actions = torch.as_tensor(arrays["actions"], device=device)
    old_log_probabilities = torch.as_tensor(arrays["log_probabilities"], device=device)
    returns_tensor = torch.as_tensor(returns, device=device)
    advantages_tensor = torch.as_tensor(advantages, device=device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    count = 0
    for _ in range(epochs):
        order = np.random.permutation(len(buffer))
        for start in range(0, len(order), minibatch_size):
            indices = torch.as_tensor(order[start:start + minibatch_size], device=device)
            new_log_probabilities, values, entropy = policy.evaluate_actions(
                observations[indices], actions[indices]
            )
            ratio = torch.exp(new_log_probabilities - old_log_probabilities[indices])
            unclipped = ratio * advantages_tensor[indices]
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages_tensor[indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns_tensor[indices] - values).pow(2).mean()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            count += 1
            metrics["loss"] += float(loss.item())
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy.mean().item())
    buffer.clear()
    return {key: value / max(count, 1) for key, value in metrics.items()}
