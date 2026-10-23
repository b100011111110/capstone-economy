from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import numpy as np
import torch

from economy_rl.config import Config
from economy_rl.envs.adapter import agent_observation, encode_actions
from economy_rl.envs.parallel import ParallelWorlds
from economy_rl.models.actor_critic import ActorCritic
from economy_rl.rl.ppo import update_policy
from economy_rl.rl.rollout import RolloutBuffer
from economy_rl.utils.data_dump import DataDump
from economy_rl.utils.history import HistoryWriter


class RunningRewardNormalizer:
    def __init__(self, clip: float = 10.0):
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0
        self.clip = clip

    def normalize(self, reward: float) -> float:
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2
        std = max(float(np.sqrt(self.var / self.count)), 1e-4)
        scaled = (reward - self.mean) / std
        return float(np.clip(scaled, -self.clip, self.clip))


def _pad_vectors(vectors: List[np.ndarray], size: int) -> np.ndarray:
    result = np.zeros((len(vectors), size), dtype=np.float32)
    for index, vector in enumerate(vectors):
        result[index, : min(size, len(vector))] = vector[:size]
    return result


def _pad_tensor(vector: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            "cuda" if config.device == "auto" and torch.cuda.is_available() else
            ("cpu" if config.device == "auto" else config.device)
        )
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.worlds = ParallelWorlds(config)
        
        # Calculate dynamic observation sizes
        planner_vectors = []
        worker_vectors = []
        for world_id in range(config.n_worlds):
            planner_vectors.append(self.worlds.get_planner_observation(world_id))
            worker_vectors.extend(self._raw_worker_vector(world_id, agent_id) for agent_id in self.worlds.worker_ids)
        self.planner_observation_size = max(len(vector) for vector in planner_vectors)
        self.worker_observation_size = max(len(vector) for vector in worker_vectors)

        if self.config.is_voting_mode:
            # 3 Competing Democratic Leader Models
            self.leaders: List[ActorCritic] = [
                ActorCritic(
                    self.planner_observation_size, self.worlds.planner_dims, config.hidden_size
                ).to(self.device)
                for _ in range(config.n_leaders)
            ]
            self.planner = self.leaders[0]
            self.leader_optimizers = [
                torch.optim.Adam(leader.parameters(), lr=config.planner_learning_rate)
                for leader in self.leaders
            ]
            self.planner_optimizer = self.leader_optimizers[0]
            self.leader_normalizers = [RunningRewardNormalizer() for _ in range(config.n_leaders)]
            self.planner_normalizer = self.leader_normalizers[0]
        else:
            # Single Social Planner Model (Normal Simulation)
            self.planner = ActorCritic(
                self.planner_observation_size, self.worlds.planner_dims, config.hidden_size
            ).to(self.device)
            self.leaders = [self.planner]
            self.planner_optimizer = torch.optim.Adam(self.planner.parameters(), lr=config.planner_learning_rate)
            self.leader_optimizers = [self.planner_optimizer]
            self.planner_normalizer = RunningRewardNormalizer()
            self.leader_normalizers = [self.planner_normalizer]

        # Worker Model
        self.worker = ActorCritic(
            self.worker_observation_size, (self.worlds.worker_action_size,), config.hidden_size
        ).to(self.device)
        self.worker_optimizer = torch.optim.Adam(self.worker.parameters(), lr=config.learning_rate)
        self.worker_normalizer = RunningRewardNormalizer()

        self.history = HistoryWriter(config.output_dir)
        self.dump = DataDump(config.output_dir)
        self.global_step = 0
        self.update_index = 0
        self.pending_planner: List[Optional[Dict[str, Any]]] = [None] * config.n_worlds
        self.episode_planner_rewards = [0.0] * config.n_worlds
        self.episode_worker_rewards = [0.0] * config.n_worlds
        self.episode_lengths = [0] * config.n_worlds
        self.episode_agent_cumulative_rewards: List[Dict[str, float]] = [
            {a_id: 0.0 for a_id in self.worlds.worker_ids}
            for _ in range(self.config.n_worlds)
        ]
        self.previous_agent_labor: List[Dict[str, float]] = [
            {a_id: 0.0 for a_id in self.worlds.worker_ids}
            for _ in range(self.config.n_worlds)
        ]
        self._saved_checkpoints: List[Path] = []

        if self.config.checkpoint_path:
            self.load_checkpoint(self.config.checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load worker
        if "worker" in checkpoint:
            try:
                worker_state = dict(checkpoint["worker"])
                cur_state = self.worker.state_dict()
                if "encoder.0.weight" in worker_state and cur_state["encoder.0.weight"].shape != worker_state["encoder.0.weight"].shape:
                    old_w = worker_state["encoder.0.weight"]
                    new_w = cur_state["encoder.0.weight"].clone()
                    min_cols = min(old_w.shape[1], new_w.shape[1])
                    new_w[:, :min_cols] = old_w[:, :min_cols]
                    worker_state["encoder.0.weight"] = new_w
                self.worker.load_state_dict(worker_state, strict=False)
            except Exception as e:
                print(f"Skipping worker checkpoint load: {e}")

        # Load leaders if present
        if "leaders" in checkpoint:
            for idx, leader_state in enumerate(checkpoint["leaders"]):
                if idx < len(self.leaders):
                    try:
                        cur_state = self.leaders[idx].state_dict()
                        if "encoder.0.weight" in leader_state and cur_state["encoder.0.weight"].shape != leader_state["encoder.0.weight"].shape:
                            old_w = leader_state["encoder.0.weight"]
                            new_w = cur_state["encoder.0.weight"].clone()
                            min_cols = min(old_w.shape[1], new_w.shape[1])
                            new_w[:, :min_cols] = old_w[:, :min_cols]
                            leader_state["encoder.0.weight"] = new_w
                        self.leaders[idx].load_state_dict(leader_state, strict=False)
                    except Exception as e:
                        print(f"Skipping leader {idx} load: {e}")
        elif "planner" in checkpoint:
            try:
                planner_state = dict(checkpoint["planner"])
                for leader in self.leaders:
                    cur_state = leader.state_dict()
                    state_to_load = dict(planner_state)
                    if "encoder.0.weight" in state_to_load and cur_state["encoder.0.weight"].shape != state_to_load["encoder.0.weight"].shape:
                        old_w = state_to_load["encoder.0.weight"]
                        new_w = cur_state["encoder.0.weight"].clone()
                        min_cols = min(old_w.shape[1], new_w.shape[1])
                        new_w[:, :min_cols] = old_w[:, :min_cols]
                        state_to_load["encoder.0.weight"] = new_w
                    leader.load_state_dict(state_to_load, strict=False)
                print("Successfully initialized all 3 leaders from planner checkpoint with warm weights!")
            except Exception as e:
                print(f"Skipping planner checkpoint load: {e}")

        self.worker_optimizer = torch.optim.Adam(self.worker.parameters(), lr=self.config.learning_rate)
        self.leader_optimizers = [
            torch.optim.Adam(leader.parameters(), lr=self.config.planner_learning_rate)
            for leader in self.leaders
        ]
        self.planner_optimizer = self.leader_optimizers[0]
        print(f"Loaded checkpoint weights with worker lr={self.config.learning_rate}, planner lr={self.config.planner_learning_rate}", flush=True)

    def _planner_input(self, world_id: int) -> torch.Tensor:
        vector = self.worlds.get_planner_observation(world_id)
        return torch.as_tensor(_pad_vectors([vector], self.planner_observation_size), device=self.device)

    def _worker_input(self, world_id: int) -> torch.Tensor:
        vectors = [
            self._raw_worker_vector(world_id, agent_id)
            for agent_id in self.worlds.worker_ids
        ]
        return torch.as_tensor(_pad_vectors(vectors, self.worker_observation_size), device=self.device)

    def _raw_worker_vector(self, world_id: int, agent_id: str) -> np.ndarray:
        base = agent_observation(self.worlds.observations(world_id), agent_id)
        return np.concatenate((base, self.worlds.policy_features(world_id)))

    def _reward(self, rewards: Dict[str, Any], agent_id: str) -> float:
        value = rewards.get(agent_id, rewards.get(str(agent_id), 0.0))
        if isinstance(value, (list, tuple, np.ndarray)):
            return float(np.asarray(value, dtype=np.float32).sum())
        return float(value)

    def collect_rollout(self) -> Dict[str, Any]:
        worker_buffer = RolloutBuffer()
        leader_buffers = [RolloutBuffer() for _ in range(self.config.n_leaders)]

        for _ in range(self.config.rollout_steps):
            for world_id in range(self.config.n_worlds):
                world_state = self.worlds.worlds[world_id]
                episode_id = world_state.episode_id
                timestep = world_state.timestep
                current_leader_id = self.worlds.current_leader_id(world_id)

                planner_input = self._planner_input(world_id)
                if self.worlds.needs_planner_action(world_id):
                    active_leader = self.leaders[current_leader_id]
                    with torch.no_grad():
                        action, log_probability, value, _ = active_leader.sample(planner_input)
                    planner_action = action[0].cpu().numpy()
                    self.worlds.set_planner_action(world_id, planner_action)
                    self.pending_planner[world_id] = {
                        "leader_id": current_leader_id,
                        "observation": planner_input[0].cpu().numpy(),
                        "action": planner_action,
                        "log_probability": float(log_probability.item()),
                        "value": float(value.item()),
                        "reward": 0.0,
                        "duration": 0,
                    }

                worker_input = self._worker_input(world_id)
                with torch.no_grad():
                    worker_action, worker_log_probability, worker_value, _ = self.worker.sample(worker_input)
                actions = encode_actions(
                    self.worlds.planner_id,
                    self.worlds.worker_ids,
                    self.worlds.planner_action(world_id),
                    worker_action[:, 0].cpu().numpy(),
                )
                planner_action_for_step = tuple(self.worlds.planner_action(world_id))
                next_observations, rewards, done, step_info = self.worlds.step(world_id, actions)

                planner_reward = self._reward(rewards, self.worlds.planner_id)
                worker_rewards = [self._reward(rewards, agent_id) for agent_id in self.worlds.worker_ids]
                worker_total_reward = float(np.sum(worker_rewards))
                worker_mean_reward = float(np.mean(worker_rewards))
                worker_action_values = worker_action[:, 0].cpu().numpy().astype(np.int64)
                worker_action_mean = float(np.mean(worker_action_values))
                worker_action_mode = int(np.bincount(worker_action_values).argmax())
                reward_gap = abs(planner_reward - worker_mean_reward) / (
                    abs(planner_reward) + abs(worker_mean_reward) + 1e-8
                )
                self.episode_planner_rewards[world_id] += planner_reward
                self.episode_worker_rewards[world_id] += worker_total_reward
                self.episode_lengths[world_id] += 1

                # Log step
                self.dump.step({
                    "global_step": self.global_step,
                    "world": world_id,
                    "episode": episode_id,
                    "timestep": timestep,
                    "planner_reward": planner_reward,
                    "worker_total_reward": worker_total_reward,
                    "worker_mean_reward": worker_mean_reward,
                    "reward_gap": reward_gap,
                    "planner_action_value": int(planner_action_for_step[0]) if planner_action_for_step else -1,
                    "worker_action_mean": worker_action_mean,
                    "worker_action_mode": worker_action_mode,
                    "done": done,
                    "planner_action": json.dumps([int(value) for value in planner_action_for_step]),
                })

                # Capture per-agent actions, efforts, resources, rewards, and loyalties
                agent_records = step_info.get("agent_records", [])
                for i, rec in enumerate(agent_records):
                    a_id = rec["agent_id"]
                    rec["global_step"] = self.global_step
                    rec["world"] = world_id
                    rec["episode"] = episode_id
                    rec["timestep"] = timestep
                    rec["action"] = int(worker_action_values[i]) if i < len(worker_action_values) else 0
                    rew = float(worker_rewards[i]) if i < len(worker_rewards) else 0.0
                    rec["step_reward"] = rew
                    self.episode_agent_cumulative_rewards[world_id][a_id] += rew
                    rec["cumulative_reward"] = self.episode_agent_cumulative_rewards[world_id][a_id]

                    prev_lab = self.previous_agent_labor[world_id].get(a_id, 0.0)
                    rec["step_labor"] = max(0.0, rec["labor"] - prev_lab)
                    self.previous_agent_labor[world_id][a_id] = rec["labor"]

                if self.config.log_agent_metrics:
                    if timestep % self.config.agent_log_interval == 0 or done:
                        self.dump.agent_steps(agent_records)

                if self.config.is_voting_mode:
                    # Log democratic election if held
                    if step_info.get("election_held") and step_info.get("election_result"):
                        res = step_info["election_result"]
                        self.dump.election({
                            "global_step": self.global_step,
                            "world": world_id,
                            "episode": episode_id,
                            "timestep": timestep,
                            "incumbent_id": current_leader_id,
                            "winner_id": res.winner_id,
                            "incumbent_reelected": res.incumbent_reelected,
                            "margin_of_victory": res.margin_of_victory,
                            "vote_counts": json.dumps(res.vote_counts),
                            "vote_shares": json.dumps([float(s) for s in res.vote_shares]),
                            "cluster_vote_shares": json.dumps(res.cluster_vote_shares.tolist()) if res.cluster_vote_shares is not None else "[]",
                        })

                    # Log loyalty state
                    if "mean_leader_loyalty" in step_info:
                        loyalties = step_info["mean_leader_loyalty"]
                        self.dump.loyalty({
                            "global_step": self.global_step,
                            "world": world_id,
                            "episode": episode_id,
                            "timestep": timestep,
                            "incumbent_id": current_leader_id,
                            "leader_0_loyalty": loyalties[0] if len(loyalties) > 0 else 0.0,
                            "leader_1_loyalty": loyalties[1] if len(loyalties) > 1 else 0.0,
                            "leader_2_loyalty": loyalties[2] if len(loyalties) > 2 else 0.0,
                            "cluster_proportions": json.dumps(step_info.get("cluster_proportions", [])),
                        })

                # Add worker transitions
                next_worker_vectors = [
                    np.concatenate((
                        agent_observation(next_observations, agent_id),
                        self.worlds.policy_features(world_id),
                    ))
                    for agent_id in self.worlds.worker_ids
                ]
                next_worker_input = torch.as_tensor(
                    _pad_vectors(next_worker_vectors, self.worker_observation_size),
                    device=self.device,
                )
                with torch.no_grad():
                    next_worker_values = self.worker.sample(
                        next_worker_input, deterministic=True
                    )[2].cpu().numpy()
                for index, agent_id in enumerate(self.worlds.worker_ids):
                    raw_rew = worker_rewards[index]
                    norm_rew = self.worker_normalizer.normalize(raw_rew) if self.config.normalize_rewards else raw_rew
                    worker_buffer.add(
                        observation=worker_input[index].cpu().numpy(),
                        action=worker_action[index].cpu().numpy(),
                        log_probability=float(worker_log_probability[index].item()),
                        value=float(worker_value[index].item()),
                        reward=norm_rew,
                        done=done,
                        next_value=0.0 if done else float(next_worker_values[index]),
                    )

                # Add leader transitions
                pending = self.pending_planner[world_id]
                if pending is not None:
                    pending["reward"] += planner_reward
                    pending["duration"] += 1
                    close_interval = done or self.worlds.needs_planner_action(world_id)
                    if close_interval:
                        l_id = pending["leader_id"]
                        next_planner = self.worlds.get_planner_observation(world_id)
                        next_padded = _pad_vectors([next_planner], self.planner_observation_size)[0]
                        with torch.no_grad():
                            next_value = 0.0 if done else float(self.leaders[l_id].sample(_pad_tensor(next_padded, self.device), deterministic=True)[2].item())
                        p_raw_rew = pending["reward"]
                        p_norm_rew = self.leader_normalizers[l_id].normalize(p_raw_rew) if self.config.normalize_rewards else p_raw_rew
                        leader_buffers[l_id].add(
                            observation=pending["observation"],
                            action=pending["action"],
                            log_probability=pending["log_probability"],
                            value=pending["value"],
                            reward=p_norm_rew,
                            done=done,
                            next_value=next_value,
                        )
                        hist_rec = {
                            "type": "planner_interval",
                            "world": world_id,
                            "duration": pending["duration"],
                            "reward": pending["reward"],
                            "done": done,
                        }
                        if self.config.is_voting_mode:
                            hist_rec["leader_id"] = l_id
                        self.history.write(hist_rec)
                        self.dump.planner_interval({
                            "global_step": self.global_step,
                            "world": world_id,
                            "episode": episode_id,
                            "duration": pending["duration"],
                            "reward": pending["reward"],
                            "done": done,
                            "planner_action": json.dumps([int(value) for value in pending["action"]]),
                        })
                        self.pending_planner[world_id] = None

                if done:
                    if self.config.log_agent_metrics:
                        self.dump.agent_episodes(agent_records)
                    self.dump.episode({
                        "global_step": self.global_step,
                        "world": world_id,
                        "episode": episode_id,
                        "length": self.episode_lengths[world_id],
                        "planner_total_reward": self.episode_planner_rewards[world_id],
                        "worker_total_reward": self.episode_worker_rewards[world_id],
                        "worker_mean_reward": self.episode_worker_rewards[world_id] / max(self.episode_lengths[world_id] * self.config.n_agents, 1),
                    })
                    self.episode_planner_rewards[world_id] = 0.0
                    self.episode_worker_rewards[world_id] = 0.0
                    self.episode_lengths[world_id] = 0
                    self.episode_agent_cumulative_rewards[world_id] = {a_id: 0.0 for a_id in self.worlds.worker_ids}
                    self.previous_agent_labor[world_id] = {a_id: 0.0 for a_id in self.worlds.worker_ids}
                self.global_step += 1

        return {"worker": worker_buffer, "leaders": leader_buffers}

    def train(self) -> None:
        for update_index in range(self.config.updates):
            buffers = self.collect_rollout()
            worker_entropy_coeff = getattr(self.config, "worker_entropy_coefficient", self.config.entropy_coefficient)
            planner_entropy_coeff = getattr(self.config, "planner_entropy_coefficient", self.config.entropy_coefficient)
            
            # 1. Update Worker Policy
            worker_metrics = update_policy(
                self.worker, self.worker_optimizer, buffers["worker"],
                self.config.gamma, self.config.gae_lambda, self.config.clip_epsilon,
                self.config.value_coefficient, worker_entropy_coeff,
                self.config.ppo_epochs, self.config.minibatch_size, self.device,
            )

            # 2. Update Democratic Leader Policies / Social Planner Policy
            if self.config.is_voting_mode:
                leader_metrics_list = []
                for k in range(self.config.n_leaders):
                    l_buf = buffers["leaders"][k]
                    if len(l_buf) > 0:
                        l_metrics = update_policy(
                            self.leaders[k], self.leader_optimizers[k], l_buf,
                            self.config.gamma, self.config.gae_lambda, self.config.clip_epsilon,
                            self.config.value_coefficient, planner_entropy_coeff,
                            self.config.ppo_epochs, self.config.minibatch_size, self.device,
                        )
                    else:
                        l_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
                    leader_metrics_list.append(l_metrics)

                primary_planner_metrics = leader_metrics_list[0]
                record = {
                    "type": "update",
                    "update": update_index,
                    "global_step": self.global_step,
                    "worker": worker_metrics,
                    "planner": primary_planner_metrics,
                    "leaders": leader_metrics_list,
                }
            else:
                p_buf = buffers["leaders"][0]
                if len(p_buf) > 0:
                    planner_metrics = update_policy(
                        self.planner, self.planner_optimizer, p_buf,
                        self.config.gamma, self.config.gae_lambda, self.config.clip_epsilon,
                        self.config.value_coefficient, planner_entropy_coeff,
                        self.config.ppo_epochs, self.config.minibatch_size, self.device,
                    )
                else:
                    planner_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
                primary_planner_metrics = planner_metrics
                record = {
                    "type": "update",
                    "update": update_index,
                    "global_step": self.global_step,
                    "worker": worker_metrics,
                    "planner": planner_metrics,
                }

            self.history.write(record)
            self.dump.update({
                "update": update_index,
                "global_step": self.global_step,
                "worker_loss": worker_metrics["loss"],
                "worker_policy_loss": worker_metrics["policy_loss"],
                "worker_value_loss": worker_metrics["value_loss"],
                "worker_entropy": worker_metrics["entropy"],
                "planner_loss": primary_planner_metrics["loss"],
                "planner_policy_loss": primary_planner_metrics["policy_loss"],
                "planner_value_loss": primary_planner_metrics["value_loss"],
                "planner_entropy": primary_planner_metrics["entropy"],
            })

            # Checkpoint saving controlled by json config
            if self.config.save_weights and self.config.checkpoint_interval > 0:
                if update_index % self.config.checkpoint_interval == 0:
                    self.save_checkpoint(update_index)
            print(record)

        self.worlds.close()

        # Save final weights if configured
        if self.config.save_weights and self.config.save_final_weights:
            if self.config.checkpoint_interval <= 0 or (self.config.updates - 1) % self.config.checkpoint_interval != 0:
                self.save_checkpoint(self.config.updates, is_final=True)

        summary_data = {
            "mode": "voting" if self.config.is_voting_mode else "normal",
            "distribution": "uniform" if self.config.is_uniform_distribution else "non_uniform",
            "world_size": self.config.world_size,
            "n_agents": self.config.n_agents,
            "n_worlds": self.config.n_worlds,
            "planner_interval": self.config.planner_interval,
            "global_steps": self.global_step,
            "updates": self.config.updates,
            "save_weights": self.config.save_weights,
            "checkpoint_interval": self.config.checkpoint_interval,
            "keep_last_n_checkpoints": self.config.keep_last_n_checkpoints,
            "delete_old_checkpoints": self.config.delete_old_checkpoints,
            "delete_checkpoints_after_training": self.config.delete_checkpoints_after_training,
            "log_agent_metrics": self.config.log_agent_metrics,
            "agent_log_interval": self.config.agent_log_interval,
            "data_directory": str(Path(self.config.output_dir) / "data"),
        }
        if self.config.is_voting_mode:
            summary_data["n_leaders"] = self.config.n_leaders
            summary_data["election_interval"] = self.config.election_interval

        self.dump.write_summary(summary_data)
        self.dump.close()
        self._write_post_training_report()

        # Delete all checkpoints after training if requested in json config
        if self.config.delete_checkpoints_after_training:
            self.delete_all_checkpoints()

    def _write_post_training_report(self) -> None:
        from economy_rl.analysis.analyze import analyze

        analysis_dir, report = analyze(
            str(Path(self.config.output_dir) / "data"),
            str(Path(self.config.output_dir) / "analysis"),
        )
        print("training_complete", flush=True)
        print("graph=%s/economy_training.png" % analysis_dir, flush=True)
        print("fit_status=%s" % report.get("fit_status", "unknown"), flush=True)
        print("fit_explanation=%s" % report.get("fit_explanation", "not available"), flush=True)

    def save_checkpoint(self, update_index: int, is_final: bool = False) -> Optional[Path]:
        if not self.config.save_weights:
            return None
        destination_dir = Path(self.config.output_dir) / "checkpoints"
        destination_dir.mkdir(parents=True, exist_ok=True)
        filename = "final.pt" if is_final else ("update_%06d.pt" % update_index)
        destination = destination_dir / filename
        payload = {
            "config": self.config.to_dict(),
            "worker": self.worker.state_dict(),
            "worker_optimizer": self.worker_optimizer.state_dict(),
            "update": update_index,
            "global_step": self.global_step,
            "mode": "voting" if self.config.is_voting_mode else "normal",
        }
        if self.config.is_voting_mode:
            payload["leaders"] = [l.state_dict() for l in self.leaders]
            payload["leader_optimizers"] = [opt.state_dict() for opt in self.leader_optimizers]
            payload["planner"] = self.leaders[0].state_dict()
            payload["planner_optimizer"] = self.leader_optimizers[0].state_dict()
        else:
            payload["planner"] = self.planner.state_dict()
            payload["planner_optimizer"] = self.planner_optimizer.state_dict()
            payload["leaders"] = [self.planner.state_dict()]
            payload["leader_optimizers"] = [self.planner_optimizer.state_dict()]

        torch.save(payload, destination)
        print(f"Saved checkpoint: {destination}", flush=True)

        if not is_final:
            self._manage_checkpoint_retention()
        return destination

    def _manage_checkpoint_retention(self) -> None:
        keep_n = self.config.keep_last_n_checkpoints
        if self.config.delete_old_checkpoints and keep_n is None:
            keep_n = 1
        if keep_n is not None and keep_n >= 0:
            checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
            if not checkpoint_dir.exists():
                return
            ckpts = sorted(checkpoint_dir.glob("update_*.pt"))
            if len(ckpts) > keep_n:
                to_delete = ckpts[:-keep_n] if keep_n > 0 else ckpts
                for old_file in to_delete:
                    try:
                        old_file.unlink()
                        print(f"Deleted old checkpoint: {old_file}", flush=True)
                    except OSError as e:
                        print(f"Warning: could not delete old checkpoint {old_file}: {e}")

    def delete_all_checkpoints(self) -> None:
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        if checkpoint_dir.exists():
            for pt_file in list(checkpoint_dir.glob("*.pt")):
                try:
                    pt_file.unlink()
                    print(f"Deleted checkpoint: {pt_file}", flush=True)
                except OSError as e:
                    print(f"Warning: could not delete checkpoint {pt_file}: {e}")
