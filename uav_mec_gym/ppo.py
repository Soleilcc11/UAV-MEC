"""PPO for the frozen mixed discrete/continuous UAV-MEC action contract.

The policy deliberately keeps ``action_mask`` out of the continuous feature
vector.  The mask is used only by the categorical target distribution, while
the UAV movement command is modelled by a tanh-squashed diagonal Normal.

This module is CPU-first.  In addition to making tests repeatable, that keeps
checkpoint/restart behaviour independent of CUDA implementation details.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from .contract import PROTOCOL_VERSION


__all__ = [
    "PPOConfig",
    "ActionSample",
    "RolloutBuffer",
    "MixedActionPPO",
    "RunningObservationNormalizer",
    "InvalidActionMaskError",
    "masked_logits",
    "masked_categorical",
    "squashed_normal_log_prob",
    "compute_gae",
]


_OBSERVATION_KEYS = frozenset(
    {"time", "delta_time", "task", "resources", "uavs", "action_mask"}
)
_CONTINUOUS_KEYS = ("time", "delta_time", "task", "resources", "uavs")


class InvalidActionMaskError(RuntimeError, ValueError):
    """The simulator supplied an action mask with no executable target."""


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters and all behavioural choices needed to resume PPO."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 10
    minibatch_size: int = 64
    rollout_steps: int = 2048
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_sizes: tuple[int, ...] = (128, 128)
    log_std_init: float = -0.5
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    normalize_observations: bool = True
    normalize_advantages: bool = True
    normalizer_epsilon: float = 1e-8
    normalizer_clip: float = 10.0
    bootstrap_truncated: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Normalise JSON-decoded checkpoint lists back to the public tuple form.
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.clip_ratio <= 0.0:
            raise ValueError("clip_ratio must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.update_epochs <= 0:
            raise ValueError("update_epochs must be positive")
        if self.minibatch_size <= 0:
            raise ValueError("minibatch_size must be positive")
        if self.rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        if self.value_coef < 0.0 or self.entropy_coef < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if not self.log_std_min <= self.log_std_init <= self.log_std_max:
            raise ValueError("log_std_init must lie inside the configured bounds")
        if self.normalizer_epsilon <= 0.0:
            raise ValueError("normalizer_epsilon must be positive")
        if self.normalizer_clip <= 0.0:
            raise ValueError("normalizer_clip must be positive")
        if self.device != "cpu":
            raise ValueError("MixedActionPPO currently guarantees reproducibility on CPU")


@dataclass(frozen=True)
class ActionSample:
    """Policy output plus the exact old-policy data PPO must retain."""

    action: dict[str, Any]
    normalized_observation: np.ndarray
    action_mask: np.ndarray
    log_prob: float
    value: float
    target_log_prob: float
    movement_log_prob: float

    @property
    def state(self) -> np.ndarray:
        """Alias used by rollout code and external tests."""

        return self.normalized_observation

    @property
    def target(self) -> int:
        return int(self.action["target"])

    @property
    def movement(self) -> np.ndarray:
        return np.asarray(self.action["movement"], dtype=np.float32)


class RunningObservationNormalizer:
    """Numerically stable running moments for flattened continuous features."""

    def __init__(self, shape: int, epsilon: float = 1e-8, clip: float = 10.0):
        if shape <= 0:
            raise ValueError("normalizer shape must be positive")
        self.shape = (int(shape),)
        self.epsilon = float(epsilon)
        self.clip = float(clip)
        self.mean = np.zeros(self.shape, dtype=np.float64)
        self.var = np.ones(self.shape, dtype=np.float64)
        self.count = 0.0

    def update(self, values: np.ndarray | Sequence[float]) -> None:
        batch = np.asarray(values, dtype=np.float64)
        if batch.shape == self.shape:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2 or batch.shape[1:] != self.shape:
            raise ValueError(
                f"normalizer expected (*, {self.shape[0]}) values, got {batch.shape}"
            )
        if batch.shape[0] == 0:
            return
        if not np.isfinite(batch).all():
            raise ValueError("normalizer cannot consume non-finite observations")

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = float(batch.shape[0])
        if self.count == 0.0:
            self.mean = batch_mean
            self.var = batch_var
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        old_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count
        new_m2 = (
            old_m2
            + batch_m2
            + np.square(delta) * self.count * batch_count / total_count
        )
        self.mean = new_mean
        self.var = new_m2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.shape != self.shape:
            raise ValueError(
                f"normalizer expected shape {self.shape}, got {array.shape}"
            )
        if self.count == 0.0:
            return array.copy()
        normalized = (array.astype(np.float64) - self.mean) / np.sqrt(
            self.var + self.epsilon
        )
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "clip": self.clip,
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        shape = tuple(int(value) for value in state["shape"])
        if shape != self.shape:
            raise ValueError(
                f"normalizer checkpoint shape {shape} does not match {self.shape}"
            )
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        if mean.shape != self.shape or var.shape != self.shape:
            raise ValueError("invalid normalizer moment shapes in checkpoint")
        count = float(state["count"])
        if count < 0.0 or (var < 0.0).any():
            raise ValueError("invalid normalizer statistics in checkpoint")
        self.epsilon = float(state.get("epsilon", self.epsilon))
        self.clip = float(state.get("clip", self.clip))
        self.mean = mean.copy()
        self.var = var.copy()
        self.count = count


def _coerce_mask(action_mask: torch.Tensor | np.ndarray, logits: torch.Tensor) -> torch.Tensor:
    mask = torch.as_tensor(action_mask, device=logits.device)
    if mask.shape != logits.shape:
        raise ValueError(
            f"action_mask shape {tuple(mask.shape)} does not match logits "
            f"shape {tuple(logits.shape)}"
        )
    if mask.dtype is not torch.bool:
        if torch.is_floating_point(mask) and not torch.isfinite(mask).all():
            raise ValueError("action_mask contains non-finite values")
        if not torch.logical_or(mask == 0, mask == 1).all():
            raise ValueError("action_mask must contain only 0/1 values")
        mask = mask.to(dtype=torch.bool)
    if mask.ndim == 0:
        raise ValueError("action_mask must have a target dimension")
    if not mask.any(dim=-1).all():
        raise InvalidActionMaskError("action_mask disables every execution target")
    return mask


def masked_logits(
    logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray
) -> torch.Tensor:
    """Return logits whose illegal entries are exactly ``-inf``.

    A row with no legal action is an invalid simulator state and raises a
    ``RuntimeError`` instead of silently selecting a fallback target.
    """

    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim == 0:
        raise ValueError("logits must have a target dimension")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must have a floating dtype")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain non-finite values")
    mask = _coerce_mask(action_mask, logits)
    return logits.masked_fill(~mask, -torch.inf)


def masked_categorical(
    logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray
) -> Categorical:
    """Categorical distribution with exactly zero illegal-action probability."""

    return Categorical(logits=masked_logits(logits, action_mask))


def squashed_normal_log_prob(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    movement: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Log-density of an *executed* tanh-squashed Normal movement.

    ``movement`` is inverted with ``atanh``.  Computing from the bounded action
    (rather than retaining the pre-tanh draw) ensures the rollout records the
    log-probability of the float32 command that was actually sent to Gymnasium.
    """

    if mean.shape != log_std.shape or movement.shape != mean.shape:
        raise ValueError("mean, log_std, and movement must have identical shapes")
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    if not torch.isfinite(mean).all() or not torch.isfinite(log_std).all():
        raise ValueError("Normal parameters must be finite")
    if not torch.isfinite(movement).all():
        raise ValueError("movement must be finite")
    if (movement < -1.0).any() or (movement > 1.0).any():
        raise ValueError("movement must lie in [-1, 1]")

    bounded = movement.clamp(-1.0 + epsilon, 1.0 - epsilon)
    pre_tanh = torch.atanh(bounded)
    distribution = Normal(mean, log_std.exp())
    base_log_prob = distribution.log_prob(pre_tanh)
    # Stable log(1 - tanh(x)^2), from the SAC appendix.
    log_jacobian = 2.0 * (
        math.log(2.0) - pre_tanh - torch.nn.functional.softplus(-2.0 * pre_tanh)
    )
    return (base_log_prob - log_jacobian).sum(dim=-1)


def compute_gae(
    rewards: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    next_values: Sequence[float] | np.ndarray,
    terminated: Sequence[bool] | np.ndarray,
    truncated: Sequence[bool] | np.ndarray,
    *,
    gamma: float = 0.99,
    discounts: Sequence[float] | np.ndarray | None = None,
    gae_lambda: float = 0.95,
    bootstrap_truncated: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute episode-safe Generalized Advantage Estimation.

    Terminations never bootstrap.  Truncations optionally bootstrap their
    one-step delta, but both flags always stop the recursive GAE carry so an
    advantage can never leak into the next episode in a combined rollout.
    """

    arrays = [
        np.asarray(rewards, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        np.asarray(next_values, dtype=np.float64),
        np.asarray(terminated, dtype=np.bool_),
        np.asarray(truncated, dtype=np.bool_),
    ]
    names = ("rewards", "values", "next_values", "terminated", "truncated")
    for name, array in zip(names, arrays):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    length = arrays[0].shape[0]
    if any(array.shape[0] != length for array in arrays[1:]):
        raise ValueError("all GAE inputs must have the same length")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    if not all(np.isfinite(array).all() for array in arrays[:3]):
        raise ValueError("GAE rewards and values must be finite")
    if np.logical_and(arrays[3], arrays[4]).any():
        raise ValueError("a transition cannot be both terminated and truncated")

    discount_array = (
        np.full(length, gamma, dtype=np.float64)
        if discounts is None
        else np.asarray(discounts, dtype=np.float64)
    )
    if discount_array.ndim != 1 or discount_array.shape[0] != length:
        raise ValueError("discounts must be one-dimensional and match rewards")
    if (
        not np.isfinite(discount_array).all()
        or (discount_array < 0.0).any()
        or (discount_array > 1.0).any()
    ):
        raise ValueError("discounts must be finite and lie in [0, 1]")

    reward_array, value_array, next_value_array, terminated_array, truncated_array = arrays
    advantages = np.zeros(length, dtype=np.float64)
    carry = 0.0
    for index in range(length - 1, -1, -1):
        can_bootstrap = not bool(terminated_array[index]) and (
            bootstrap_truncated or not bool(truncated_array[index])
        )
        delta = (
            reward_array[index]
            + discount_array[index]
            * next_value_array[index]
            * float(can_bootstrap)
            - value_array[index]
        )
        same_episode = not (
            bool(terminated_array[index]) or bool(truncated_array[index])
        )
        carry = (
            delta
            + discount_array[index]
            * gae_lambda
            * float(same_episode)
            * carry
        )
        advantages[index] = carry
    returns = advantages + value_array
    return advantages.astype(np.float32), returns.astype(np.float32)


class RolloutBuffer:
    """On-policy transitions, including the old joint action log-probability."""

    _FIELDS = (
        "states",
        "action_masks",
        "targets",
        "movements",
        "log_probs",
        "values",
        "rewards",
        "discounts",
        "next_values",
        "terminated",
        "truncated",
    )

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states: list[np.ndarray] = []
        self.action_masks: list[np.ndarray] = []
        self.targets: list[int] = []
        self.movements: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.discounts: list[float] = []
        self.next_values: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        *,
        state: np.ndarray,
        action_mask: np.ndarray,
        target: int,
        movement: np.ndarray,
        log_prob: float,
        value: float,
        reward: float,
        discount: float,
        next_value: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        scalar_values = (log_prob, value, reward, discount, next_value)
        if not all(math.isfinite(float(item)) for item in scalar_values):
            raise ValueError("rollout scalar values must be finite")
        if not 0.0 <= float(discount) <= 1.0:
            raise ValueError("rollout discount must lie in [0, 1]")
        state_array = np.asarray(state, dtype=np.float32)
        mask_array = np.asarray(action_mask, dtype=np.bool_)
        movement_array = np.asarray(movement, dtype=np.float32)
        if state_array.ndim != 1 or mask_array.ndim != 1 or movement_array.ndim != 2:
            raise ValueError("invalid rollout tensor rank")
        if target < 0 or target >= mask_array.shape[0]:
            raise ValueError("target is outside action_mask")
        if not mask_array[target]:
            raise ValueError("cannot store a masked execution target")
        if movement_array.shape[1:] != (3,):
            raise ValueError("movement must have shape (number_of_uavs, 3)")
        if not np.isfinite(state_array).all() or not np.isfinite(movement_array).all():
            raise ValueError("rollout arrays must be finite")
        if (movement_array < -1.0).any() or (movement_array > 1.0).any():
            raise ValueError("movement must lie in [-1, 1]")

        self.states.append(state_array.copy())
        self.action_masks.append(mask_array.copy())
        self.targets.append(int(target))
        self.movements.append(movement_array.copy())
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.discounts.append(float(discount))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def compute_advantages(
        self, config: PPOConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            self.rewards,
            self.values,
            self.next_values,
            self.terminated,
            self.truncated,
            gamma=config.gamma,
            discounts=self.discounts,
            gae_lambda=config.gae_lambda,
            bootstrap_truncated=config.bootstrap_truncated,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            field: [
                item.copy() if isinstance(item, np.ndarray) else item
                for item in getattr(self, field)
            ]
            for field in self._FIELDS
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.clear()
        lengths = {len(state[field]) for field in self._FIELDS}
        if len(lengths) != 1:
            raise ValueError("rollout checkpoint fields have inconsistent lengths")
        for items in zip(*(state[field] for field in self._FIELDS)):
            row = dict(zip(self._FIELDS, items))
            self.add(
                state=row["states"],
                action_mask=row["action_masks"],
                target=int(row["targets"]),
                movement=row["movements"],
                log_prob=float(row["log_probs"]),
                value=float(row["values"]),
                reward=float(row["rewards"]),
                discount=float(row["discounts"]),
                next_value=float(row["next_values"]),
                terminated=bool(row["terminated"]),
                truncated=bool(row["truncated"]),
            )


def _make_mlp(input_dim: int, hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for size in hidden_sizes:
        layers.extend((nn.Linear(previous, size), nn.Tanh()))
        previous = size
    return nn.Sequential(*layers)


class _MixedActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        target_count: int,
        movement_dim: int,
        config: PPOConfig,
    ) -> None:
        super().__init__()
        self.trunk = _make_mlp(observation_dim, config.hidden_sizes)
        feature_dim = config.hidden_sizes[-1]
        self.target_head = nn.Linear(feature_dim, target_count)
        self.movement_mean_head = nn.Linear(feature_dim, movement_dim)
        self.movement_log_std = nn.Parameter(
            torch.full((movement_dim,), config.log_std_init, dtype=torch.float32)
        )
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        logits = self.target_head(features)
        mean = self.movement_mean_head(features)
        log_std = self.movement_log_std.clamp(
            self.log_std_min, self.log_std_max
        ).expand_as(mean)
        return logits, mean, log_std


class _Critic(nn.Module):
    def __init__(
        self, observation_dim: int, hidden_sizes: tuple[int, ...]
    ) -> None:
        super().__init__()
        self.trunk = _make_mlp(observation_dim, hidden_sizes)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.trunk(state)).squeeze(-1)


class MixedActionPPO:
    """PPO agent for ``Dict(target=Discrete, movement=Box)`` actions."""

    name = "mixed_action_ppo"
    checkpoint_format_version = 3

    def __init__(
        self,
        number_of_uavs: int,
        config: PPOConfig | None = None,
        seed: int = 0,
    ) -> None:
        if number_of_uavs <= 0:
            raise ValueError("number_of_uavs must be positive")
        self.number_of_uavs = int(number_of_uavs)
        self.target_count = 2 + self.number_of_uavs
        self.observation_dim = 2 + 7 + 2 * 3 + self.number_of_uavs * 8
        self.movement_dim = self.number_of_uavs * 3
        self.config = config if config is not None else PPOConfig()
        self.device = torch.device(self.config.device)
        self.seed = int(seed)

        self._seed_global_rngs(self.seed)

        # Public, agent-local generators.  Torch distributions do not accept a
        # generator, so sampling below uses these generators explicitly.
        self.python_rng = random.Random(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(self.seed)

        # fork_rng makes seeded parameter initialisation reproducible without
        # consuming or replacing the caller's process-global torch RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.actor = _MixedActor(
                self.observation_dim,
                self.target_count,
                self.movement_dim,
                self.config,
            ).to(self.device)
            self.critic = _Critic(
                self.observation_dim, self.config.hidden_sizes
            ).to(self.device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )
        self.observation_normalizer = RunningObservationNormalizer(
            self.observation_dim,
            epsilon=self.config.normalizer_epsilon,
            clip=self.config.normalizer_clip,
        )
        # Convenient short alias, retained in checkpoints under the long name.
        self.normalizer = self.observation_normalizer
        self.buffer = RolloutBuffer()
        self.training_steps = 0
        self.update_count = 0
        # Environment episode seeds are a separate stream from policy-action
        # sampling.  Training must never call ``reset`` on a restored agent,
        # because doing so would discard the checkpointed action RNG state.
        self.training_seed_start: int | None = None
        self.training_episode_count = 0
        self._legacy_next_training_seed: int | None = None

    @property
    def next_training_seed(self) -> int | None:
        """Next environment seed in the persisted training seed schedule."""

        if self.training_seed_start is None:
            return None
        return self.training_seed_start + self.training_episode_count

    def reserve_training_episode_seed(self, training_seed_start: int) -> int:
        """Reserve the next non-repeating environment seed for training.

        The schedule is initialized on the first call and then persisted in
        checkpoints.  A resume command must supply the same seed-series start;
        this catches accidental restarts that would otherwise silently reuse
        training episodes.  Version-1 checkpoints predate the explicit cursor,
        so their last policy seed is conservatively treated as already used.
        """

        requested_start = int(training_seed_start)
        if self.training_seed_start is None:
            self.training_seed_start = requested_start
            if self._legacy_next_training_seed is not None:
                inferred_count = self._legacy_next_training_seed - requested_start
                if inferred_count < 0:
                    raise ValueError(
                        "training_seed_start is later than the next seed inferred "
                        "from this legacy checkpoint"
                    )
                self.training_episode_count = inferred_count
                self._legacy_next_training_seed = None
        elif requested_start != self.training_seed_start:
            raise ValueError(
                "training_seed_start does not match the checkpointed training "
                f"schedule ({requested_start} != {self.training_seed_start})"
            )

        seed = self.next_training_seed
        if seed is None:  # pragma: no cover - guarded by initialization above.
            raise AssertionError("training seed schedule was not initialized")
        self.training_episode_count += 1
        return seed

    @staticmethod
    def masked_logits(
        logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray
    ) -> torch.Tensor:
        return masked_logits(logits, action_mask)

    @staticmethod
    def masked_categorical(
        logits: torch.Tensor, action_mask: torch.Tensor | np.ndarray
    ) -> Categorical:
        return masked_categorical(logits, action_mask)

    @staticmethod
    def compute_gae(
        rewards: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray,
        next_values: Sequence[float] | np.ndarray,
        terminated: Sequence[bool] | np.ndarray,
        truncated: Sequence[bool] | np.ndarray,
        *,
        gamma: float = 0.99,
        discounts: Sequence[float] | np.ndarray | None = None,
        gae_lambda: float = 0.95,
        bootstrap_truncated: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=gamma,
            discounts=discounts,
            gae_lambda=gae_lambda,
            bootstrap_truncated=bootstrap_truncated,
        )

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
                f"observation does not match frozen Dict contract; "
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
            raise ValueError(
                f"observation['action_mask'] has shape {raw_mask.shape}, "
                f"expected {expected_shapes['action_mask']}"
            )
        if not np.logical_or(raw_mask == 0, raw_mask == 1).all():
            raise ValueError("observation['action_mask'] must contain only 0/1 values")
        action_mask = raw_mask.astype(np.bool_, copy=True)

        continuous = np.concatenate(
            [arrays[key].reshape(-1) for key in _CONTINUOUS_KEYS]
        ).astype(np.float32, copy=False)
        if continuous.shape != (self.observation_dim,):
            raise AssertionError("internal observation dimension mismatch")
        return continuous, action_mask

    def _prepare_observation(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        update_normalizer: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        continuous, action_mask = self._flatten_observation(observation)
        if self.config.normalize_observations:
            if update_normalizer:
                self.observation_normalizer.update(continuous)
            state = self.observation_normalizer.normalize(continuous)
        else:
            state = continuous.copy()
        return state, action_mask

    def _policy_parameters(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor(states)

    def _evaluate_actions(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        targets: torch.Tensor,
        movements: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, means, log_stds = self._policy_parameters(states)
        target_distribution = masked_categorical(logits, action_masks)
        target_log_probs = target_distribution.log_prob(targets)
        flat_movements = movements.reshape(-1, self.movement_dim)
        movement_log_probs = squashed_normal_log_prob(
            means, log_stds, flat_movements
        )
        joint_log_probs = target_log_probs + movement_log_probs
        # The base-Normal entropy is a low-variance proxy for the squashed
        # entropy; target entropy is exact.  This does not enter PPO's ratio.
        entropy = target_distribution.entropy() + Normal(
            means, log_stds.exp()
        ).entropy().sum(dim=-1)
        return joint_log_probs, entropy, self.critic(states)

    def sample_action(
        self,
        observation: Mapping[str, np.ndarray],
        deterministic: bool = False,
        update_normalizer: bool = False,
    ) -> ActionSample:
        state, action_mask = self._prepare_observation(
            observation, update_normalizer=update_normalizer
        )
        if not action_mask.any():
            raise InvalidActionMaskError(
                "action_mask disables every execution target"
            )
        state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
        mask_tensor = torch.from_numpy(action_mask).to(self.device).unsqueeze(0)
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            logits, mean, log_std = self._policy_parameters(state_tensor)
            target_distribution = masked_categorical(logits, mask_tensor)
            if deterministic:
                target_tensor = masked_logits(logits, mask_tensor).argmax(dim=-1)
                pre_tanh = mean
            else:
                # Explicit local generators make rollout sampling checkpointable.
                target_tensor = torch.multinomial(
                    target_distribution.probs,
                    num_samples=1,
                    replacement=True,
                    generator=self.torch_rng,
                ).squeeze(-1)
                noise = torch.randn(
                    mean.shape,
                    dtype=mean.dtype,
                    device=mean.device,
                    generator=self.torch_rng,
                )
                pre_tanh = mean + log_std.exp() * noise

            # Convert first, then reconstruct a tensor from the executable
            # float32 command for the old movement log-probability.
            movement = (
                torch.tanh(pre_tanh)
                .clamp(-1.0 + 1e-6, 1.0 - 1e-6)
                .reshape(self.number_of_uavs, 3)
                .cpu()
                .numpy()
                .astype(np.float32, copy=True)
            )
            executed_movement = (
                torch.from_numpy(movement.reshape(1, -1)).to(self.device)
            )
            target_log_prob = target_distribution.log_prob(target_tensor)
            movement_log_prob = squashed_normal_log_prob(
                mean, log_std, executed_movement
            )
            value = self.critic(state_tensor)

        action = {"target": int(target_tensor.item()), "movement": movement}
        target_log_prob_value = float(target_log_prob.item())
        movement_log_prob_value = float(movement_log_prob.item())
        return ActionSample(
            action=action,
            normalized_observation=state.copy(),
            action_mask=action_mask.copy(),
            log_prob=target_log_prob_value + movement_log_prob_value,
            value=float(value.item()),
            target_log_prob=target_log_prob_value,
            movement_log_prob=movement_log_prob_value,
        )

    def act(
        self,
        observation: Mapping[str, np.ndarray],
        deterministic: bool = True,
    ) -> dict[str, Any]:
        """Return only the frozen Gymnasium action, suitable for evaluation."""

        return self.sample_action(
            observation,
            deterministic=deterministic,
            update_normalizer=False,
        ).action

    def reset(self, seed: int) -> None:
        """Reset agent-local action-sampling streams for evaluation only.

        Training intentionally keeps a continuous RNG stream across environment
        episodes and checkpoint reloads; ``train_ppo`` therefore never calls
        this method.
        """

        self.seed = int(seed)
        self._seed_global_rngs(self.seed)
        self.python_rng.seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        self.torch_rng.manual_seed(self.seed)

    @staticmethod
    def _seed_global_rngs(seed: int) -> None:
        """Seed every framework RNG required by the reproducibility contract."""

        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def store_transition(
        self,
        sample: ActionSample,
        reward: float,
        next_observation: Mapping[str, np.ndarray],
        terminated: bool,
        truncated: bool,
        discount: float | None = None,
    ) -> None:
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        state = np.asarray(sample.normalized_observation, dtype=np.float32)
        action_mask = np.asarray(sample.action_mask, dtype=np.bool_)
        if state.shape != (self.observation_dim,):
            raise ValueError("ActionSample state does not match this agent")
        if action_mask.shape != (self.target_count,):
            raise ValueError("ActionSample action_mask does not match this agent")
        movement = np.asarray(sample.action["movement"], dtype=np.float32)
        if movement.shape != (self.number_of_uavs, 3):
            raise ValueError("ActionSample movement does not match this agent")

        next_state, _ = self._prepare_observation(
            next_observation, update_normalizer=False
        )
        next_state_tensor = torch.from_numpy(next_state).to(self.device).unsqueeze(0)
        self.critic.eval()
        with torch.no_grad():
            next_value = float(self.critic(next_state_tensor).item())

        self.buffer.add(
            state=state,
            action_mask=action_mask,
            target=int(sample.action["target"]),
            movement=movement,
            log_prob=float(sample.log_prob),
            value=float(sample.value),
            reward=float(reward),
            discount=self.config.gamma if discount is None else float(discount),
            next_value=next_value,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self.training_steps += 1

    def update(self) -> dict[str, float] | None:
        """Update from every buffered transition, including a short tail batch."""

        transition_count = len(self.buffer)
        if transition_count == 0:
            return None

        advantages, returns = self.buffer.compute_advantages(self.config)
        if self.config.normalize_advantages and transition_count > 1:
            advantage_mean = float(advantages.mean())
            advantage_std = float(advantages.std())
            if advantage_std > 1e-8:
                advantages = (advantages - advantage_mean) / (advantage_std + 1e-8)

        states = torch.from_numpy(np.stack(self.buffer.states)).to(self.device)
        action_masks = torch.from_numpy(np.stack(self.buffer.action_masks)).to(
            self.device
        )
        targets = torch.tensor(
            self.buffer.targets, dtype=torch.long, device=self.device
        )
        movements = torch.from_numpy(np.stack(self.buffer.movements)).to(self.device)
        old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=self.device
        )
        advantage_tensor = torch.from_numpy(advantages).to(self.device)
        return_tensor = torch.from_numpy(returns).to(self.device)

        self.actor.train()
        self.critic.train()
        metric_sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        metric_weight = 0
        batch_size = min(self.config.minibatch_size, transition_count)

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(
                transition_count, generator=self.torch_rng, device="cpu"
            )
            for start in range(0, transition_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                new_log_probs, entropy, predicted_values = self._evaluate_actions(
                    states[indices],
                    action_masks[indices],
                    targets[indices],
                    movements[indices],
                )
                log_ratio = new_log_probs - old_log_probs[indices]
                ratio = log_ratio.exp()
                unclipped = ratio * advantage_tensor[indices]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantage_tensor[indices]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(
                    predicted_values, return_tensor[indices]
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                count = int(indices.numel())
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)
                        .float()
                        .mean()
                    )
                values = {
                    "loss": loss,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy_mean,
                    "approx_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                }
                for key, value in values.items():
                    metric_sums[key] += float(value.detach().item()) * count
                metric_weight += count

        self.update_count += 1
        self.buffer.clear()
        metrics = {
            key: total / metric_weight for key, total in metric_sums.items()
        }
        metrics.update(
            {
                "rollout_steps": float(transition_count),
                "training_steps": float(self.training_steps),
                "update_count": float(self.update_count),
                "advantage_mean": float(advantages.mean()),
                "return_mean": float(returns.mean()),
            }
        )
        return metrics

    def _rng_state(self) -> dict[str, Any]:
        return {
            "python_global": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_global": torch.get_rng_state(),
            "agent_python": self.python_rng.getstate(),
            "agent_numpy": self.numpy_rng.bit_generator.state,
            "agent_torch": self.torch_rng.get_state(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        """Save enough state to continue training and sampling exactly."""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        actor_state = self.actor.state_dict()
        critic_state = self.critic.state_dict()
        optimizer_state = self.optimizer.state_dict()
        rng_state = self._rng_state()
        checkpoint = {
            "format_version": self.checkpoint_format_version,
            "algorithm": self.name,
            "protocol_version": PROTOCOL_VERSION,
            "number_of_uavs": self.number_of_uavs,
            "config": asdict(self.config),
            "actor_state_dict": actor_state,
            "critic_state_dict": critic_state,
            "optimizer_state_dict": optimizer_state,
            # Short aliases make the file straightforward to inspect and retain
            # compatibility with conventional PyTorch checkpoint naming.
            "actor": actor_state,
            "critic": critic_state,
            "optimizer": optimizer_state,
            "training_steps": self.training_steps,
            "update_count": self.update_count,
            "seed": self.seed,
            "training_seed_start": self.training_seed_start,
            "training_episode_count": self.training_episode_count,
            "observation_normalizer": self.observation_normalizer.state_dict(),
            "rollout_buffer": self.buffer.state_dict(),
            "rng_state": rng_state,
            "python_rng_state": rng_state["python_global"],
            "numpy_rng_state": rng_state["numpy_global"],
            "agent_rng_state": {
                "python": rng_state["agent_python"],
                "numpy": rng_state["agent_numpy"],
                "torch": rng_state["agent_torch"],
            },
            "torch_rng_state": rng_state["torch_global"],
        }
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "MixedActionPPO":
        """Construct an agent from a checkpoint and restore all RNG streams."""

        checkpoint_path = Path(path)
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch < 2.0 has no weights_only argument.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("invalid PPO checkpoint")
        format_version = int(checkpoint.get("format_version", 1))
        if (
            format_version != cls.checkpoint_format_version
            or checkpoint.get("algorithm") != cls.name
            or checkpoint.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError(
                "unsupported PPO checkpoint format; GymBridge 1.1 "
                "checkpoints are intentionally incompatible with protocol 1.2"
            )

        config = PPOConfig(**dict(checkpoint["config"]))
        agent = cls(
            number_of_uavs=int(checkpoint["number_of_uavs"]),
            config=config,
            seed=int(checkpoint["seed"]),
        )
        actor_state = (
            checkpoint["actor_state_dict"]
            if "actor_state_dict" in checkpoint
            else checkpoint["actor"]
        )
        critic_state = (
            checkpoint["critic_state_dict"]
            if "critic_state_dict" in checkpoint
            else checkpoint["critic"]
        )
        optimizer_state = (
            checkpoint["optimizer_state_dict"]
            if "optimizer_state_dict" in checkpoint
            else checkpoint["optimizer"]
        )
        agent.actor.load_state_dict(actor_state)
        agent.critic.load_state_dict(critic_state)
        agent.optimizer.load_state_dict(optimizer_state)
        agent.training_steps = int(checkpoint["training_steps"])
        agent.update_count = int(checkpoint["update_count"])
        raw_training_seed_start = checkpoint.get("training_seed_start")
        agent.training_seed_start = (
            None
            if raw_training_seed_start is None
            else int(raw_training_seed_start)
        )
        agent.training_episode_count = int(
            checkpoint.get("training_episode_count", 0)
        )
        if agent.training_episode_count < 0:
            raise ValueError("invalid training episode count in checkpoint")
        if (
            agent.training_seed_start is None
            and agent.training_episode_count != 0
        ):
            raise ValueError("checkpoint has a seed cursor without a seed start")
        agent.observation_normalizer.load_state_dict(
            checkpoint["observation_normalizer"]
        )
        if "rollout_buffer" in checkpoint:
            agent.buffer.load_state_dict(checkpoint["rollout_buffer"])

        rng_state = checkpoint.get("rng_state")
        if rng_state is None:
            # Compatibility with the explicit top-level representation.
            agent_rng = checkpoint["agent_rng_state"]
            rng_state = {
                "python_global": checkpoint["python_rng_state"],
                "numpy_global": checkpoint["numpy_rng_state"],
                "torch_global": checkpoint["torch_rng_state"],
                "agent_python": agent_rng["python"],
                "agent_numpy": agent_rng["numpy"],
                "agent_torch": agent_rng["torch"],
            }
        random.setstate(rng_state["python_global"])
        np.random.set_state(rng_state["numpy_global"])
        torch.set_rng_state(rng_state["torch_global"])
        agent.python_rng.setstate(rng_state["agent_python"])
        agent.numpy_rng = np.random.default_rng()
        agent.numpy_rng.bit_generator.state = rng_state["agent_numpy"]
        agent.torch_rng.set_state(rng_state["agent_torch"])
        return agent
