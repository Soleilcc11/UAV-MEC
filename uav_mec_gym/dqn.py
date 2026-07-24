"""Masked DQN baseline on the real UAV-MEC GymBridge.

DQN controls the discrete execution target and deliberately emits zero UAV
movement because vanilla DQN has no continuous-action head.  It therefore
shares observations, rewards, seeds, task arrivals, and the Java simulator with
the mixed-action algorithms while exposing this action-capability limitation in
its recorded configuration.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contract import PROTOCOL_VERSION
from .ppo import InvalidActionMaskError, RunningObservationNormalizer


__all__ = ["DQNConfig", "DQNActionSample", "MaskedDQN"]


_OBSERVATION_KEYS = frozenset(
    {"time", "delta_time", "task", "resources", "uavs", "action_mask"}
)
_CONTINUOUS_KEYS = ("time", "delta_time", "task", "resources", "uavs")


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 100_000
    learning_starts: int = 1_000
    target_update_interval: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 10_000
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
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if self.target_update_interval <= 0 or self.epsilon_decay_steps <= 0:
            raise ValueError("update and decay intervals must be positive")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not self.hidden_sizes or any(size <= 0 for size in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer sizes")
        if self.normalizer_epsilon <= 0.0 or self.normalizer_clip <= 0.0:
            raise ValueError("normalizer parameters must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.device != "cpu":
            raise ValueError("MaskedDQN currently guarantees reproducibility on CPU")


@dataclass(frozen=True)
class DQNActionSample:
    action: dict[str, Any]
    raw_observation: np.ndarray
    action_mask: np.ndarray
    epsilon: float


class _ReplayBuffer:
    _ARRAY_NAMES = (
        "states", "action_masks", "actions", "rewards", "discounts", "next_states",
        "next_action_masks", "terminated", "truncated",
    )

    def __init__(self, capacity: int, observation_dim: int, target_count: int) -> None:
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.target_count = int(target_count)
        self.states = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.action_masks = np.zeros((capacity, target_count), dtype=np.bool_)
        self.actions = np.zeros(capacity, dtype=np.int64)
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
        action: int,
        reward: float,
        discount: float,
        next_state: np.ndarray,
        next_action_mask: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if terminated and truncated:
            raise ValueError("a transition cannot be both terminated and truncated")
        if state.shape != (self.observation_dim,) or next_state.shape != (
            self.observation_dim,
        ):
            raise ValueError("replay observation shape mismatch")
        if action_mask.shape != (self.target_count,) or next_action_mask.shape != (
            self.target_count,
        ):
            raise ValueError("replay action-mask shape mismatch")
        if action < 0 or action >= self.target_count or not action_mask[action]:
            raise ValueError("replay action is not enabled by action_mask")
        if not math.isfinite(float(reward)):
            raise ValueError("replay reward must be finite")
        if not math.isfinite(float(discount)) or not 0.0 <= discount <= 1.0:
            raise ValueError("replay discount must be finite and lie in [0, 1]")
        index = self.position
        self.states[index] = state
        self.action_masks[index] = action_mask
        self.actions[index] = action
        self.rewards[index] = reward
        self.discounts[index] = discount
        self.next_states[index] = next_state
        self.next_action_masks[index] = next_action_mask
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if batch_size <= 0 or batch_size > self.size:
            raise ValueError("invalid replay batch_size")
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return {name: getattr(self, name)[indices].copy() for name in self._ARRAY_NAMES}

    def state_dict(self) -> dict[str, Any]:
        retained = self.capacity if self.size == self.capacity else self.size
        return {
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "target_count": self.target_count,
            "position": self.position,
            "size": self.size,
            "arrays": {
                name: getattr(self, name)[:retained].copy()
                for name in self._ARRAY_NAMES
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.capacity, self.observation_dim, self.target_count)
        actual = (
            int(state["capacity"]), int(state["observation_dim"]),
            int(state["target_count"]),
        )
        if actual != expected:
            raise ValueError("DQN replay checkpoint dimensions do not match")
        size = int(state["size"])
        position = int(state["position"])
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("invalid DQN replay checkpoint cursor")
        retained = self.capacity if size == self.capacity else size
        for name in self._ARRAY_NAMES:
            target = getattr(self, name)
            saved = np.asarray(state["arrays"][name])
            if saved.shape != target[:retained].shape:
                raise ValueError(f"invalid DQN replay checkpoint array: {name}")
            target.fill(0)
            target[:retained] = saved.astype(target.dtype, copy=False)
        self.size = size
        self.position = position


class _QNetwork(nn.Module):
    def __init__(self, observation_dim: int, target_count: int,
            hidden_sizes: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = observation_dim
        for size in hidden_sizes:
            layers.extend((nn.Linear(previous, size), nn.ReLU()))
            previous = size
        layers.append(nn.Linear(previous, target_count))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class MaskedDQN:
    name = "masked_dqn_zero_movement"
    checkpoint_format_version = 2

    def __init__(self, number_of_uavs: int,
            config: DQNConfig | None = None, seed: int = 0) -> None:
        if number_of_uavs <= 0:
            raise ValueError("number_of_uavs must be positive")
        self.number_of_uavs = int(number_of_uavs)
        self.target_count = 2 + self.number_of_uavs
        self.observation_dim = 2 + 7 + 2 * 3 + self.number_of_uavs * 8
        self.config = config if config is not None else DQNConfig()
        self.device = torch.device(self.config.device)
        self.seed = int(seed)
        self._seed_global_rngs(self.seed)
        self.python_rng = random.Random(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.q_network = _QNetwork(
                self.observation_dim, self.target_count, self.config.hidden_sizes
            ).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.learning_rate
        )
        self.normalizer = RunningObservationNormalizer(
            self.observation_dim,
            epsilon=self.config.normalizer_epsilon,
            clip=self.config.normalizer_clip,
        )
        self.replay_buffer = _ReplayBuffer(
            self.config.replay_capacity, self.observation_dim, self.target_count
        )
        self.training_steps = 0
        self.update_count = 0
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
            raise ValueError("training_seed_start does not match checkpoint")
        seed = self.next_training_seed
        if seed is None:
            raise AssertionError("training seed schedule was not initialized")
        self.training_episode_count += 1
        return seed

    def _flatten_observation(self, observation: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if frozenset(observation.keys()) != _OBSERVATION_KEYS:
            raise ValueError("observation does not match the UAV-MEC Dict contract")
        expected = {
            "time": (1,), "delta_time": (1,), "task": (7,),
            "resources": (2, 3),
            "uavs": (self.number_of_uavs, 8),
        }
        arrays: list[np.ndarray] = []
        for key in _CONTINUOUS_KEYS:
            value = np.asarray(observation[key], dtype=np.float32)
            if value.shape != expected[key] or not np.isfinite(value).all():
                raise ValueError(f"invalid observation field: {key}")
            arrays.append(value.reshape(-1))
        raw_mask = np.asarray(observation["action_mask"])
        if raw_mask.shape != (self.target_count,) or not np.logical_or(
            raw_mask == 0, raw_mask == 1
        ).all():
            raise ValueError("invalid observation action_mask")
        state = np.concatenate(arrays).astype(np.float32, copy=False)
        return state, raw_mask.astype(np.bool_, copy=True)

    def _normalise(self, states: np.ndarray) -> np.ndarray:
        values = np.asarray(states, dtype=np.float32)
        if not self.config.normalize_observations or self.normalizer.count == 0.0:
            return values.copy()
        result = (values.astype(np.float64) - self.normalizer.mean) / np.sqrt(
            self.normalizer.var + self.normalizer.epsilon
        )
        return np.clip(result, -self.normalizer.clip, self.normalizer.clip).astype(np.float32)

    def epsilon(self) -> float:
        fraction = min(1.0, self.training_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def sample_action(self, observation: Mapping[str, np.ndarray],
            deterministic: bool = False, update_normalizer: bool = False) -> DQNActionSample:
        raw_state, mask = self._flatten_observation(observation)
        if not mask.any():
            raise InvalidActionMaskError("action_mask disables every execution target")
        if update_normalizer and self.config.normalize_observations:
            self.normalizer.update(raw_state)
        exploration = 0.0 if deterministic else self.epsilon()
        if self.numpy_rng.random() < exploration:
            target = int(self.numpy_rng.choice(np.flatnonzero(mask)))
        else:
            state = torch.from_numpy(self._normalise(raw_state)).to(self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state).squeeze(0)
                q_values[~torch.from_numpy(mask).to(self.device)] = -torch.inf
                target = int(q_values.argmax().item())
        return DQNActionSample(
            action={
                "target": target,
                "movement": np.zeros((self.number_of_uavs, 3), dtype=np.float32),
            },
            raw_observation=raw_state.copy(),
            action_mask=mask.copy(),
            epsilon=float(exploration),
        )

    def act(self, observation: Mapping[str, np.ndarray], deterministic: bool = True) -> dict[str, Any]:
        return self.sample_action(observation, deterministic=deterministic).action

    def store_transition(self, sample: DQNActionSample, reward: float,
            next_observation: Mapping[str, np.ndarray], terminated: bool,
            truncated: bool, discount: float | None = None) -> None:
        next_state, next_mask = self._flatten_observation(next_observation)
        self.replay_buffer.add(
            state=np.asarray(sample.raw_observation, dtype=np.float32),
            action_mask=np.asarray(sample.action_mask, dtype=np.bool_),
            action=int(sample.action["target"]), reward=float(reward),
            discount=self.config.gamma if discount is None else float(discount),
            next_state=next_state, next_action_mask=next_mask,
            terminated=bool(terminated), truncated=bool(truncated),
        )
        self.training_steps += 1

    def update(self) -> dict[str, float] | None:
        if self.training_steps < self.config.learning_starts or len(
            self.replay_buffer
        ) < self.config.batch_size:
            return None
        batch = self.replay_buffer.sample(self.config.batch_size, self.numpy_rng)
        states = torch.from_numpy(self._normalise(batch["states"])).to(self.device)
        next_states = torch.from_numpy(self._normalise(batch["next_states"])).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        rewards = torch.from_numpy(batch["rewards"]).to(self.device)
        discounts = torch.from_numpy(batch["discounts"]).to(self.device)
        next_masks = torch.from_numpy(batch["next_action_masks"]).to(self.device)
        terminated = torch.from_numpy(batch["terminated"]).to(self.device)
        truncated = torch.from_numpy(batch["truncated"]).to(self.device)

        with torch.no_grad():
            has_legal = next_masks.any(dim=-1)
            if (~has_legal & ~(terminated | truncated)).any():
                raise InvalidActionMaskError("non-terminal replay transition has no legal target")
            safe_masks = next_masks.clone()
            safe_masks[~has_legal, 0] = True
            online_next = self.q_network(next_states).masked_fill(~safe_masks, -torch.inf)
            next_actions = online_next.argmax(dim=-1)
            target_next = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            will_bootstrap = (
                ~terminated & has_legal
                if self.config.bootstrap_truncated
                else ~(terminated | truncated)
            )
            target = rewards + discounts * will_bootstrap.to(
                rewards.dtype
            ) * target_next

        current = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(current, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        return {
            "training_step": float(self.training_steps),
            "update": float(self.update_count),
            "buffer_size": float(len(self.replay_buffer)),
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "q_mean": float(current.mean().item()),
            "target_q_mean": float(target.mean().item()),
            "epsilon": self.epsilon(),
        }

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._seed_global_rngs(self.seed)
        self.python_rng.seed(self.seed)
        self.numpy_rng = np.random.default_rng(self.seed)

    @staticmethod
    def _seed_global_rngs(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": self.checkpoint_format_version,
            "algorithm": self.name,
            "protocol_version": PROTOCOL_VERSION,
            "number_of_uavs": self.number_of_uavs,
            "seed": self.seed,
            "config": asdict(self.config),
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "training_steps": self.training_steps,
            "update_count": self.update_count,
            "training_seed_start": self.training_seed_start,
            "training_episode_count": self.training_episode_count,
            "rng": {
                "python_global": random.getstate(),
                "numpy_global": np.random.get_state(),
                "torch_global": torch.random.get_rng_state(),
                "python_local": self.python_rng.getstate(),
                "numpy_local": copy.deepcopy(self.numpy_rng.bit_generator.state),
            },
        }, destination)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "MaskedDQN":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if (
            payload.get("format_version") != cls.checkpoint_format_version
            or payload.get("algorithm") != cls.name
            or payload.get("protocol_version") != PROTOCOL_VERSION
        ):
            raise ValueError(
                "unsupported DQN checkpoint; protocol 1.1 checkpoints "
                "cannot be loaded by GymBridge 1.2"
            )
        agent = cls(
            int(payload["number_of_uavs"]),
            config=DQNConfig(**payload["config"]), seed=int(payload["seed"]),
        )
        agent.q_network.load_state_dict(payload["q_network"])
        agent.target_network.load_state_dict(payload["target_network"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.normalizer.load_state_dict(payload["normalizer"])
        agent.replay_buffer.load_state_dict(payload["replay_buffer"])
        agent.training_steps = int(payload["training_steps"])
        agent.update_count = int(payload["update_count"])
        start = payload.get("training_seed_start")
        agent.training_seed_start = None if start is None else int(start)
        agent.training_episode_count = int(payload.get("training_episode_count", 0))
        rng = payload["rng"]
        random.setstate(rng["python_global"])
        np.random.set_state(rng["numpy_global"])
        torch.random.set_rng_state(rng["torch_global"])
        agent.python_rng.setstate(rng["python_local"])
        agent.numpy_rng.bit_generator.state = copy.deepcopy(rng["numpy_local"])
        return agent
