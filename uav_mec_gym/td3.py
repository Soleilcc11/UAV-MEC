"""Auditable TD3 for the frozen mixed UAV-MEC action contract.

The original TD3 algorithm assumes a continuous action vector.  This module
uses a parameterised-action extension for ``Dict(target, movement)``:

* the actor emits masked target logits and a tanh-bounded movement command;
* environment interaction executes the legal target argmax;
* the critic receives a target one-hot vector plus the movement command;
* actor gradients use a hard straight-through masked softmax for the target;
* TD3 target-policy smoothing is applied only to the continuous movement.

The implementation is CPU-first and checkpoints every source of stochasticity,
the replay buffer, normalisation statistics, target networks, optimisers, and
the environment-training-seed cursor.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contract import PROTOCOL_VERSION
from .ppo import (
    InvalidActionMaskError,
    RunningObservationNormalizer,
    masked_logits,
)


__all__ = [
    "TD3Config",
    "TD3ActionSample",
    "ReplayBuffer",
    "MixedActionTD3",
]


_OBSERVATION_KEYS = frozenset(
    {"time", "delta_time", "task", "resources", "uavs", "action_mask"}
)
_CONTINUOUS_KEYS = ("time", "delta_time", "task", "resources", "uavs")


@dataclass(frozen=True)
class TD3Config:
    """Hyperparameters and behavioural choices required for exact resume."""

    gamma: float = 0.99
    tau: float = 0.005
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    discrete_exploration: float = 0.1
    target_temperature: float = 1.0
    hidden_sizes: tuple[int, ...] = (256, 256)
    normalize_observations: bool = True
    normalizer_epsilon: float = 1e-8
    normalizer_clip: float = 10.0
    bootstrap_truncated: bool = True
    max_grad_norm: float = 10.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        if self.actor_learning_rate <= 0.0 or self.critic_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        for name, value in (
            ("policy_noise", self.policy_noise),
            ("noise_clip", self.noise_clip),
            ("exploration_noise", self.exploration_noise),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= self.discrete_exploration <= 1.0:
            raise ValueError("discrete_exploration must be in [0, 1]")
        if self.target_temperature <= 0.0:
            raise ValueError("target_temperature must be positive")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")
        if self.normalizer_epsilon <= 0.0 or self.normalizer_clip <= 0.0:
            raise ValueError("normalizer parameters must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.device != "cpu":
            raise ValueError("MixedActionTD3 currently guarantees reproducibility on CPU")


@dataclass(frozen=True)
class TD3ActionSample:
    """Executable action and the raw state used to produce it."""

    action: dict[str, Any]
    raw_observation: np.ndarray
    action_mask: np.ndarray
    warmup_random: bool

    @property
    def state(self) -> np.ndarray:
        return self.raw_observation

    @property
    def target(self) -> int:
        return int(self.action["target"])

    @property
    def movement(self) -> np.ndarray:
        return np.asarray(self.action["movement"], dtype=np.float32)


class ReplayBuffer:
    """Fixed-capacity ring buffer with compact, exact checkpoint state."""

    _ARRAY_NAMES = (
        "states",
        "action_masks",
        "targets",
        "movements",
        "rewards",
        "discounts",
        "next_states",
        "next_action_masks",
        "terminated",
        "truncated",
    )

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        target_count: int,
        movement_dim: int,
    ) -> None:
        if min(capacity, observation_dim, target_count, movement_dim) <= 0:
            raise ValueError("replay dimensions must be positive")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.target_count = int(target_count)
        self.movement_dim = int(movement_dim)
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.action_masks = np.zeros((capacity, target_count), dtype=np.bool_)
        self.targets = np.zeros(capacity, dtype=np.int64)
        self.movements = np.zeros((capacity, movement_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.next_action_masks = np.zeros((capacity, target_count), dtype=np.bool_)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        *,
        state: np.ndarray,
        action_mask: np.ndarray,
        target: int,
        movement: np.ndarray,
        reward: float,
        discount: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action_mask = np.asarray(action_mask, dtype=np.bool_)
        next_action_mask = np.asarray(next_action_mask, dtype=np.bool_)
        movement = np.asarray(movement, dtype=np.float32).reshape(-1)
        if state.shape != (self.observation_dim,) or next_state.shape != (
            self.observation_dim,
        ):
            raise ValueError("replay observation shape mismatch")
        if action_mask.shape != (self.target_count,) or next_action_mask.shape != (
            self.target_count,
        ):
            raise ValueError("replay action-mask shape mismatch")
        if movement.shape != (self.movement_dim,):
            raise ValueError("replay movement shape mismatch")
        if target < 0 or target >= self.target_count or not action_mask[target]:
            raise ValueError("replay target is not enabled by action_mask")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise ValueError("replay observations must be finite")
        if not np.isfinite(movement).all() or (np.abs(movement) > 1.0).any():
            raise ValueError("replay movement must be finite and in [-1, 1]")
        if not math.isfinite(float(reward)):
            raise ValueError("replay reward must be finite")
        if not math.isfinite(float(discount)) or not 0.0 <= discount <= 1.0:
            raise ValueError("replay discount must be finite and lie in [0, 1]")

        index = self.position
        self.states[index] = state
        self.action_masks[index] = action_mask
        self.targets[index] = int(target)
        self.movements[index] = movement
        self.rewards[index] = float(reward)
        self.discounts[index] = float(discount)
        self.next_states[index] = next_state
        self.next_action_masks[index] = next_action_mask
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError("invalid replay batch_size")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {
            name: getattr(self, name)[indices].copy() for name in self._ARRAY_NAMES
        }

    def state_dict(self) -> dict[str, Any]:
        retained = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "target_count": self.target_count,
            "movement_dim": self.movement_dim,
            "position": self.position,
            "size": self.size,
            "arrays": {
                name: getattr(self, name)[:retained].copy()
                for name in self._ARRAY_NAMES
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        dimensions = (
            int(state["capacity"]),
            int(state["observation_dim"]),
            int(state["target_count"]),
            int(state["movement_dim"]),
        )
        expected = (
            self.capacity,
            self.observation_dim,
            self.target_count,
            self.movement_dim,
        )
        if dimensions != expected:
            raise ValueError(f"replay checkpoint dimensions {dimensions} != {expected}")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid replay checkpoint cursor")
        if size < self.capacity and position != size:
            raise ValueError("partially filled replay cursor must equal size")
        retained = self.capacity if size == self.capacity else size
        arrays = state["arrays"]
        for name in self._ARRAY_NAMES:
            saved = np.asarray(arrays[name])
            target = getattr(self, name)
            if saved.shape != target[:retained].shape:
                raise ValueError(f"invalid replay checkpoint array: {name}")
            target.fill(0)
            target[:retained] = saved.astype(target.dtype, copy=False)
        self.size = size
        self.position = position


def _make_mlp(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.ReLU()))
        previous = size
    return nn.Sequential(*layers)


class _Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        target_count: int,
        movement_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.trunk = _make_mlp(observation_dim, hidden_sizes)
        self.target_head = nn.Linear(hidden_sizes[-1], target_count)
        self.movement_head = nn.Linear(hidden_sizes[-1], movement_dim)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        return self.target_head(features), torch.tanh(self.movement_head(features))


class _QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        target_count: int,
        movement_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.trunk = _make_mlp(
            observation_dim + target_count + movement_dim, hidden_sizes
        )
        self.value = nn.Linear(hidden_sizes[-1], 1)

    def forward(
        self,
        state: torch.Tensor,
        target_vector: torch.Tensor,
        movement: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((state, target_vector, movement), dim=-1)
        return self.value(self.trunk(inputs)).squeeze(-1)


class _TwinCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        target_count: int,
        movement_dim: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.q1 = _QNetwork(
            observation_dim, target_count, movement_dim, hidden_sizes
        )
        self.q2 = _QNetwork(
            observation_dim, target_count, movement_dim, hidden_sizes
        )

    def forward(
        self,
        state: torch.Tensor,
        target_vector: torch.Tensor,
        movement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.q1(state, target_vector, movement),
            self.q2(state, target_vector, movement),
        )

    def q1_value(
        self,
        state: torch.Tensor,
        target_vector: torch.Tensor,
        movement: torch.Tensor,
    ) -> torch.Tensor:
        return self.q1(state, target_vector, movement)


class MixedActionTD3:
    """TD3 with a masked categorical target and continuous UAV movement."""

    name = "mixed_action_td3"
    checkpoint_format_version = 2

    def __init__(
        self,
        number_of_uavs: int,
        config: TD3Config | None = None,
        seed: int = 0,
    ) -> None:
        if number_of_uavs <= 0:
            raise ValueError("number_of_uavs must be positive")
        self.number_of_uavs = int(number_of_uavs)
        self.target_count = 2 + self.number_of_uavs
        self.observation_dim = 2 + 7 + 2 * 3 + self.number_of_uavs * 8
        self.movement_dim = self.number_of_uavs * 3
        self.config = config if config is not None else TD3Config()
        self.device = torch.device(self.config.device)
        self.seed = int(seed)

        self._seed_global_rngs(self.seed)
        self.python_rng = random.Random(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(self.seed)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor = _Actor(
                self.observation_dim,
                self.target_count,
                self.movement_dim,
                self.config.hidden_sizes,
            ).to(self.device)
            self.critic = _TwinCritic(
                self.observation_dim,
                self.target_count,
                self.movement_dim,
                self.config.hidden_sizes,
            ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for module in (self.target_actor, self.target_critic):
            module.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.config.critic_learning_rate
        )
        self.observation_normalizer = RunningObservationNormalizer(
            self.observation_dim,
            epsilon=self.config.normalizer_epsilon,
            clip=self.config.normalizer_clip,
        )
        self.normalizer = self.observation_normalizer
        self.replay_buffer = ReplayBuffer(
            self.config.replay_capacity,
            self.observation_dim,
            self.target_count,
            self.movement_dim,
        )
        self.training_steps = 0
        self.update_count = 0
        self.actor_update_count = 0
        self.training_seed_start: int | None = None
        self.training_episode_count = 0

    @property
    def next_training_seed(self) -> int | None:
        if self.training_seed_start is None:
            return None
        return self.training_seed_start + self.training_episode_count

    def reserve_training_episode_seed(self, training_seed_start: int) -> int:
        requested = int(training_seed_start)
        if self.training_seed_start is None:
            self.training_seed_start = requested
        elif self.training_seed_start != requested:
            raise ValueError(
                "training_seed_start does not match the checkpointed training "
                f"schedule ({requested} != {self.training_seed_start})"
            )
        seed = self.next_training_seed
        if seed is None:  # pragma: no cover
            raise AssertionError("training seed schedule was not initialized")
        self.training_episode_count += 1
        return seed

    def _flatten_observation(
        self, observation: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        keys = frozenset(observation.keys())
        if keys != _OBSERVATION_KEYS:
            missing = sorted(_OBSERVATION_KEYS - keys)
            extra = sorted(keys - _OBSERVATION_KEYS)
            raise ValueError(
                "observation does not match frozen Dict contract; "
                f"missing={missing}, extra={extra}"
            )
        expected_shapes = {
            "time": (1,),
            "delta_time": (1,),
            "task": (7,),
            "resources": (2, 3),
            "uavs": (self.number_of_uavs, 8),
            "action_mask": (self.target_count,),
        }
        arrays: dict[str, np.ndarray] = {}
        for key in _CONTINUOUS_KEYS:
            array = np.asarray(observation[key], dtype=np.float32)
            if array.shape != expected_shapes[key]:
                raise ValueError(
                    f"observation[{key!r}] has shape {array.shape}, "
                    f"expected {expected_shapes[key]}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"observation[{key!r}] contains non-finite values")
            arrays[key] = array
        raw_mask = np.asarray(observation["action_mask"])
        if raw_mask.shape != expected_shapes["action_mask"]:
            raise ValueError("observation action_mask shape mismatch")
        if not np.logical_or(raw_mask == 0, raw_mask == 1).all():
            raise ValueError("observation action_mask must contain only 0/1 values")
        action_mask = raw_mask.astype(np.bool_, copy=True)
        continuous = np.concatenate(
            [arrays[key].reshape(-1) for key in _CONTINUOUS_KEYS]
        ).astype(np.float32, copy=False)
        if continuous.shape != (self.observation_dim,):
            raise AssertionError("internal observation dimension mismatch")
        return continuous, action_mask

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if not self.config.normalize_observations or self.normalizer.count == 0.0:
            return values.copy()
        normalized = (values.astype(np.float64) - self.normalizer.mean) / np.sqrt(
            self.normalizer.var + self.normalizer.epsilon
        )
        return np.clip(
            normalized, -self.normalizer.clip, self.normalizer.clip
        ).astype(np.float32)

    @staticmethod
    def _hard_target_vector(
        logits: torch.Tensor, action_masks: torch.Tensor
    ) -> torch.Tensor:
        legal_logits = masked_logits(logits, action_masks)
        indices = legal_logits.argmax(dim=-1)
        return F.one_hot(indices, num_classes=logits.shape[-1]).to(logits.dtype)

    def _straight_through_target_vector(
        self, logits: torch.Tensor, action_masks: torch.Tensor
    ) -> torch.Tensor:
        legal_logits = masked_logits(logits, action_masks)
        probabilities = torch.softmax(
            legal_logits / self.config.target_temperature, dim=-1
        )
        hard = F.one_hot(
            legal_logits.argmax(dim=-1), num_classes=logits.shape[-1]
        ).to(logits.dtype)
        return hard + probabilities - probabilities.detach()

    def sample_action(
        self,
        observation: Mapping[str, np.ndarray],
        deterministic: bool = False,
        update_normalizer: bool = False,
    ) -> TD3ActionSample:
        raw_state, action_mask = self._flatten_observation(observation)
        if not action_mask.any():
            raise InvalidActionMaskError("action_mask disables every execution target")
        if update_normalizer and self.config.normalize_observations:
            self.normalizer.update(raw_state)
        state = self._normalise(raw_state)
        warmup_random = bool(
            not deterministic and self.training_steps < self.config.learning_starts
        )

        if warmup_random:
            legal_targets = np.flatnonzero(action_mask)
            target = int(self.numpy_rng.choice(legal_targets))
            movement = self.numpy_rng.uniform(
                -1.0, 1.0, size=self.movement_dim
            ).astype(np.float32)
        else:
            state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
            mask_tensor = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
            self.actor.eval()
            with torch.no_grad():
                logits, movement_tensor = self.actor(state_tensor)
                target = int(
                    masked_logits(logits, mask_tensor).argmax(dim=-1).item()
                )
                movement = movement_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            if not deterministic:
                if self.numpy_rng.random() < self.config.discrete_exploration:
                    target = int(self.numpy_rng.choice(np.flatnonzero(action_mask)))
                if self.config.exploration_noise > 0.0:
                    movement = movement + self.numpy_rng.normal(
                        0.0, self.config.exploration_noise, size=self.movement_dim
                    ).astype(np.float32)
                movement = np.clip(movement, -1.0, 1.0).astype(np.float32)

        return TD3ActionSample(
            action={
                "target": target,
                "movement": movement.reshape(self.number_of_uavs, 3).copy(),
            },
            raw_observation=raw_state.copy(),
            action_mask=action_mask.copy(),
            warmup_random=warmup_random,
        )

    def act(
        self,
        observation: Mapping[str, np.ndarray],
        deterministic: bool = True,
    ) -> dict[str, Any]:
        return self.sample_action(
            observation,
            deterministic=deterministic,
            update_normalizer=False,
        ).action

    def reset(self, seed: int) -> None:
        """Reset local action RNGs for evaluation; training never calls this."""

        self.seed = int(seed)
        self._seed_global_rngs(self.seed)
        self.python_rng.seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.torch_rng.manual_seed(self.seed)

    @staticmethod
    def _seed_global_rngs(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def store_transition(
        self,
        sample: TD3ActionSample,
        reward: float,
        next_observation: Mapping[str, np.ndarray],
        terminated: bool,
        truncated: bool,
        discount: float | None = None,
    ) -> None:
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        state = np.asarray(sample.raw_observation, dtype=np.float32)
        mask = np.asarray(sample.action_mask, dtype=np.bool_)
        if state.shape != (self.observation_dim,) or mask.shape != (
            self.target_count,
        ):
            raise ValueError("TD3ActionSample does not match this agent")
        next_state, next_mask = self._flatten_observation(next_observation)
        self.replay_buffer.add(
            state=state,
            action_mask=mask,
            target=int(sample.action["target"]),
            movement=np.asarray(sample.action["movement"], dtype=np.float32),
            reward=float(reward),
            discount=self.config.gamma if discount is None else float(discount),
            next_state=next_state,
            next_action_mask=next_mask,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self.training_steps += 1

    def update(self) -> dict[str, float] | None:
        if (
            self.training_steps < self.config.learning_starts
            or len(self.replay_buffer) < self.config.batch_size
        ):
            return None
        batch = self.replay_buffer.sample(self.config.batch_size, self.numpy_rng)
        states = torch.from_numpy(self._normalise(batch["states"])).to(self.device)
        next_states = torch.from_numpy(self._normalise(batch["next_states"])).to(
            self.device
        )
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
            # Natural terminal observations intentionally expose an all-zero
            # action mask.  Their bootstrap multiplier is zero, so choose a
            # harmless placeholder solely to keep the batched actor call
            # well-defined.  A non-terminal all-zero mask remains an error.
            next_has_legal_target = next_masks.any(dim=-1)
            if (~next_has_legal_target & ~(terminated | truncated)).any():
                raise InvalidActionMaskError(
                    "non-terminal replay transition has no legal next target"
                )
            if self.config.bootstrap_truncated:
                will_bootstrap = ~terminated & next_has_legal_target
            else:
                will_bootstrap = ~(terminated | truncated)
            safe_next_masks = next_masks.clone()
            safe_next_masks[~next_has_legal_target, 0] = True
            next_logits, next_movements = self.target_actor(next_states)
            next_target_vectors = self._hard_target_vector(
                next_logits, safe_next_masks
            )
            if self.config.policy_noise > 0.0:
                noise = torch.randn(
                    next_movements.shape,
                    dtype=next_movements.dtype,
                    generator=self.torch_rng,
                ).to(self.device)
                noise = (noise * self.config.policy_noise).clamp(
                    -self.config.noise_clip, self.config.noise_clip
                )
                next_movements = (next_movements + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(
                next_states, next_target_vectors, next_movements
            )
            q_target = rewards + discounts * will_bootstrap.to(
                dtype=rewards.dtype
            ) * torch.minimum(target_q1, target_q2)

        current_q1, current_q2 = self.critic(
            states, target_vectors, movements
        )
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(
            current_q2, q_target
        )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.max_grad_norm
        )
        self.critic_optimizer.step()
        self.update_count += 1

        actor_loss_value = math.nan
        actor_grad_norm_value = math.nan
        actor_updated = self.update_count % self.config.policy_delay == 0
        if actor_updated:
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
            self._soft_update(self.critic, self.target_critic)
            actor_loss_value = float(actor_loss.item())
            actor_grad_norm_value = float(actor_grad_norm)

        return {
            "training_step": float(self.training_steps),
            "critic_update": float(self.update_count),
            "actor_update": float(self.actor_update_count),
            "actor_updated": float(actor_updated),
            "buffer_size": float(len(self.replay_buffer)),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "critic_grad_norm": float(critic_grad_norm),
            "actor_grad_norm": actor_grad_norm_value,
            "target_q_mean": float(q_target.mean().item()),
            "q1_mean": float(current_q1.mean().item()),
            "q2_mean": float(current_q2.mean().item()),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        with torch.no_grad():
            for source_parameter, target_parameter in zip(
                source.parameters(), target.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.config.tau)
                target_parameter.add_(source_parameter, alpha=self.config.tau)

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python_global": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_global": torch.random.get_rng_state(),
            "python_local": self.python_rng.getstate(),
            "numpy_local": copy.deepcopy(self.numpy_rng.bit_generator.state),
            "torch_local": self.torch_rng.get_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": self.checkpoint_format_version,
                "algorithm": self.name,
                "protocol_version": PROTOCOL_VERSION,
                "number_of_uavs": self.number_of_uavs,
                "seed": self.seed,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "normalizer": self.normalizer.state_dict(),
                "replay_buffer": self.replay_buffer.state_dict(),
                "training_steps": self.training_steps,
                "update_count": self.update_count,
                "actor_update_count": self.actor_update_count,
                "training_seed_start": self.training_seed_start,
                "training_episode_count": self.training_episode_count,
                "rng": self._rng_state(),
            },
            destination,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "MixedActionTD3":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != cls.checkpoint_format_version
            or payload.get("algorithm") != cls.name
            or payload.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError(
                "unsupported TD3 checkpoint; protocol 1.1 checkpoints "
                "cannot be loaded by GymBridge 1.2"
            )
        agent = cls(
            int(payload["number_of_uavs"]),
            config=TD3Config(**payload["config"]),
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
