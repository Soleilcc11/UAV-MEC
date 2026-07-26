"""Masked parameterized-action DDPG for the real UAV-MEC GymBridge.

The actor jointly emits masked execution-target logits and bounded UAV
movement.  A hard straight-through target vector keeps the discrete branch
executable while preserving actor gradients.  Unlike TD3, this implementation
uses one critic estimate, updates the actor every critic step, and applies no
target-policy smoothing.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .contract import PROTOCOL_VERSION
from .ppo import InvalidActionMaskError
from .td3 import MixedActionTD3, TD3Config


__all__ = ["DDPGConfig", "MixedActionDDPG"]


@dataclass(frozen=True)
class DDPGConfig(TD3Config):
    """DDPG hyperparameters, including parameterized discrete exploration."""

    policy_delay: int = 1
    policy_noise: float = 0.0
    noise_clip: float = 0.0


class MixedActionDDPG(MixedActionTD3):
    """Single-critic mixed-action DDPG with action masking."""

    name = "masked_parameterized_action_ddpg"

    def __init__(
        self,
        number_of_uavs: int,
        config: DDPGConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(
            number_of_uavs,
            config=config if config is not None else DDPGConfig(),
            seed=seed,
        )

    def update(self) -> dict[str, float] | None:
        if (
            self.training_steps < self.config.learning_starts
            or len(self.replay_buffer) < self.config.batch_size
        ):
            return None
        batch = self.replay_buffer.sample(self.config.batch_size, self.numpy_rng)
        states = torch.from_numpy(self._normalise(batch["states"])).to(self.device)
        next_states = torch.from_numpy(
            self._normalise(batch["next_states"])
        ).to(self.device)
        action_masks = torch.from_numpy(batch["action_masks"]).to(self.device)
        next_masks = torch.from_numpy(batch["next_action_masks"]).to(self.device)
        targets = torch.from_numpy(batch["targets"]).to(self.device)
        target_vectors = F.one_hot(
            targets, num_classes=self.target_count
        ).to(dtype=torch.float32)
        movements = torch.from_numpy(batch["movements"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        discounts = torch.from_numpy(batch["discounts"]).to(self.device)
        terminated = torch.from_numpy(batch["terminated"]).to(self.device)
        truncated = torch.from_numpy(batch["truncated"]).to(self.device)

        with torch.no_grad():
            next_has_legal_target = next_masks.any(dim=-1)
            if (~next_has_legal_target & ~(terminated | truncated)).any():
                raise InvalidActionMaskError(
                    "non-terminal replay transition has no legal next target"
                )
            will_bootstrap = (
                ~terminated & next_has_legal_target
                if self.config.bootstrap_truncated
                else ~(terminated | truncated)
            )
            safe_next_masks = next_masks.clone()
            safe_next_masks[~next_has_legal_target, 0] = True
            next_logits, next_movements = self.target_actor(next_states)
            next_target_vectors = self._hard_target_vector(
                next_logits, safe_next_masks
            )
            target_q = self.target_critic.q1_value(
                next_states, next_target_vectors, next_movements
            )
            q_target = rewards + discounts * will_bootstrap.to(
                dtype=rewards.dtype
            ) * target_q

        current_q = self.critic.q1_value(states, target_vectors, movements)
        critic_loss = F.mse_loss(current_q, q_target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.q1.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.update_count += 1

        actor_logits, actor_movements = self.actor(states)
        actor_targets = self._straight_through_target_vector(
            actor_logits, action_masks
        )
        actor_loss = -self.critic.q1_value(
            states, actor_targets, actor_movements
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.config.max_grad_norm
        )
        self.actor_optimizer.step()
        self.actor_update_count += 1
        self._soft_update(self.actor, self.target_actor)
        self._soft_update(self.critic.q1, self.target_critic.q1)

        return {
            "training_step": float(self.training_steps),
            "critic_update": float(self.update_count),
            "actor_update": float(self.actor_update_count),
            "actor_updated": 1.0,
            "buffer_size": float(len(self.replay_buffer)),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "critic_grad_norm": float(critic_grad_norm),
            "actor_grad_norm": float(actor_grad_norm),
            "target_q_mean": float(q_target.mean().item()),
            "q_mean": float(current_q.mean().item()),
        }

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "MixedActionDDPG":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != cls.checkpoint_format_version
            or payload.get("algorithm") != cls.name
            or payload.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError(
                "unsupported DDPG checkpoint; protocol 1.1 checkpoints "
                "cannot be loaded by GymBridge 1.2"
            )
        agent = cls(
            int(payload["number_of_uavs"]),
            config=DDPGConfig(**payload["config"]),
            seed=int(payload["seed"]),
        )
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.target_actor.load_state_dict(payload["target_actor"])
        agent.target_critic.load_state_dict(payload["target_critic"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        agent.normalizer.load_state_dict(payload["normalizer"])
        agent.replay_buffer.load_state_dict(payload["replay_buffer"])
        agent.training_steps = int(payload["training_steps"])
        agent.update_count = int(payload["update_count"])
        agent.actor_update_count = int(payload["actor_update_count"])
        start = payload.get("training_seed_start")
        agent.training_seed_start = None if start is None else int(start)
        agent.training_episode_count = int(payload.get("training_episode_count", 0))

        rng = payload["rng"]
        random.setstate(rng["python_global"])
        np.random.set_state(rng["numpy_global"])
        torch.random.set_rng_state(rng["torch_global"])
        agent.python_rng.setstate(rng["python_local"])
        agent.numpy_rng.bit_generator.state = copy.deepcopy(rng["numpy_local"])
        agent.torch_rng.set_state(rng["torch_local"])
        return agent
