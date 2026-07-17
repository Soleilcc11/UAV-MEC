from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces


Observation = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class RewardComponents:
    success: float
    latency_ratio: float
    ue_energy_ratio: float
    uav_energy_ratio: float
    constraint_violations: float


@dataclass(frozen=True)
class BackendStep:
    observation: Observation
    reward_components: RewardComponents
    reward: float
    terminated: bool
    truncated: bool
    metrics: Mapping[str, Any]


class GymBackend(Protocol):
    def reset(self, seed: int | None) -> tuple[Observation, Mapping[str, Any]]: ...

    def step(self, target: int, movement: np.ndarray) -> BackendStep: ...

    def close(self) -> None: ...


class UAVMECGymEnv(gym.Env[dict[str, np.ndarray], dict[str, Any]]):
    """Gymnasium contract wrapper around an event-driven simulator backend."""

    metadata = {"render_modes": []}

    def __init__(self, number_of_uavs: int, backend: GymBackend):
        super().__init__()
        if number_of_uavs <= 0:
            raise ValueError("number_of_uavs must be positive")
        self.number_of_uavs = number_of_uavs
        self.backend = backend
        backend_uav_count = getattr(backend, "number_of_uavs", number_of_uavs)
        if backend_uav_count != number_of_uavs:
            raise ValueError("Backend UAV count does not match the Gymnasium environment")
        target_count = 3 + number_of_uavs
        self.action_space = spaces.Dict(
            {
                "target": spaces.Discrete(target_count),
                "movement": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(number_of_uavs, 3),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = spaces.Dict(
            {
                "time": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "task": spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32),
                "resources": spaces.Box(0.0, 1.0, shape=(3, 3), dtype=np.float32),
                "uavs": spaces.Box(
                    0.0, 1.0, shape=(number_of_uavs, 8), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(target_count),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.backend.reset(seed)
        return self._validate_observation(observation), dict(info)

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("Action does not satisfy the frozen Gymnasium contract")
        target = int(action["target"])
        movement = np.asarray(action["movement"], dtype=np.float32)
        result = self.backend.step(target, movement)
        observation = self._validate_observation(result.observation)
        reward = float(result.reward)
        if not np.isfinite(reward):
            raise ValueError("Backend returned a non-finite reward")
        info = dict(result.metrics)
        info["reward_components"] = {
            "success": result.reward_components.success,
            "latency_ratio": result.reward_components.latency_ratio,
            "ue_energy_ratio": result.reward_components.ue_energy_ratio,
            "uav_energy_ratio": result.reward_components.uav_energy_ratio,
            "constraint_violations": result.reward_components.constraint_violations,
        }
        info["settled_in_transition"] = int(
            result.metrics.get("settled_in_transition", 0)
        )
        return observation, reward, result.terminated, result.truncated, info

    @staticmethod
    def calculate_reward(components: RewardComponents) -> float:
        """Reference calculation for a single settled task.

        Production GymBridge 1.1 supplies the authoritative interval reward,
        which can aggregate several concurrent task settlements.
        """
        reward = (
            float(np.clip(components.success, 0.0, 1.0))
            - 0.35 * min(max(components.latency_ratio, 0.0), 2.0)
            - 0.15 * min(max(components.ue_energy_ratio, 0.0), 2.0)
            - 0.20 * min(max(components.uav_energy_ratio, 0.0), 2.0)
            - 0.30 * min(max(components.constraint_violations, 0.0), 1.0)
        )
        return float(np.clip(reward, -1.0, 1.0))

    def _validate_observation(self, observation: Observation) -> dict[str, np.ndarray]:
        normalized = {
            "time": np.asarray(observation["time"], dtype=np.float32),
            "task": np.asarray(observation["task"], dtype=np.float32),
            "resources": np.asarray(observation["resources"], dtype=np.float32),
            "uavs": np.asarray(observation["uavs"], dtype=np.float32),
            "action_mask": np.asarray(observation["action_mask"], dtype=np.int8),
        }
        if not self.observation_space.contains(normalized):
            raise ValueError("Backend observation does not satisfy the frozen contract")
        return normalized

    def close(self) -> None:
        self.backend.close()
