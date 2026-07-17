"""Reproducible baselines and evaluation utilities for the real GymBridge."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .backend import JavaGymBridgeBackend
from .contract import UAVMECGymEnv


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]: ...


class RandomMaskedPolicy:
    name = "random_masked"

    def __init__(self, number_of_uavs: int) -> None:
        self.number_of_uavs = number_of_uavs
        self._rng = np.random.default_rng()

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        enabled = np.flatnonzero(observation["action_mask"] > 0)
        if enabled.size == 0:
            raise RuntimeError("GymBridge observation has no enabled execution target")
        return {
            "target": int(self._rng.choice(enabled)),
            "movement": self._rng.uniform(
                -1.0, 1.0, size=(self.number_of_uavs, 3)
            ).astype(np.float32),
        }


class MinimumEstimatedDelayPolicy:
    """Choose the enabled target with the smallest observable transfer proxy."""

    name = "minimum_estimated_delay"

    def __init__(self, number_of_uavs: int) -> None:
        self.number_of_uavs = number_of_uavs

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        mask = observation["action_mask"] > 0
        scores = np.full(mask.shape, np.inf, dtype=np.float64)
        # Resource observation column 2 is the normalized network-delay proxy.
        if mask.size > 1 and mask[1]:
            scores[1] = float(observation["resources"][1, 2])
        if mask.size > 2 and mask[2]:
            scores[2] = float(observation["resources"][2, 2])
        for uav_id, uav in enumerate(observation["uavs"]):
            target = 3 + uav_id
            if target < mask.size and mask[target]:
                scores[target] = float(uav[6] + uav[7])
        if not np.isfinite(scores).any():
            raise RuntimeError("GymBridge observation has no enabled execution target")
        return {
            "target": int(np.argmin(scores)),
            "movement": np.zeros((self.number_of_uavs, 3), dtype=np.float32),
        }


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    seed: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    settled_tasks: int
    total_tasks: int
    simulation_time: float
    successful_tasks: int
    reward_component_sums: dict[str, float]


def evaluate_policy(env: gym.Env, policy: Policy, seeds: list[int]) -> list[EpisodeResult]:
    results: list[EpisodeResult] = []
    for seed in seeds:
        policy.reset(seed)
        observation, _ = env.reset(seed=seed)
        total_reward = 0.0
        steps = 0
        terminated = truncated = False
        info: dict[str, Any] = {}
        component_sums: dict[str, float] = {}
        successful_tasks = 0
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            total_reward += float(reward)
            steps += 1
            for key, value in info["reward_components"].items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
            successful_tasks += int(info["reward_components"]["success"] > 0.5)
        results.append(EpisodeResult(
            policy=policy.name,
            seed=seed,
            steps=steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            settled_tasks=int(info["settled_tasks"]),
            total_tasks=int(info["total_tasks"]),
            simulation_time=float(info["simulation_time"]),
            successful_tasks=successful_tasks,
            reward_component_sums=component_sums,
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12346)
    parser.add_argument("--uavs", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    backend = JavaGymBridgeBackend(
        port=args.port, expected_number_of_uavs=args.uavs
    )
    env = UAVMECGymEnv(args.uavs, backend)
    policies: list[Policy] = [
        RandomMaskedPolicy(args.uavs),
        MinimumEstimatedDelayPolicy(args.uavs),
    ]
    try:
        rows = [
            asdict(result)
            for policy in policies
            for result in evaluate_policy(env, policy, args.seeds)
        ]
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
