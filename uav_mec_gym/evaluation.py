"""Reproducible baselines and evaluation utilities for the real GymBridge."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import gymnasium as gym
import numpy as np

from .backend import JavaGymBridgeBackend
from .contract import UAVMECGymEnv


PHYSICAL_STEP_METRICS = (
    "latency_seconds",
    "deadline_seconds",
    "ue_energy_joules",
    "uav_energy_joules",
    "constraint_violations",
)


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
    physical_metric_sums: dict[str, float] = field(default_factory=dict)
    physical_metric_means: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSummary:
    count: int
    mean: float
    std: float
    ci95_lower: float
    ci95_upper: float


@dataclass(frozen=True)
class PairedSeedSummary:
    """Candidate-minus-reference differences on the exact same seed set."""

    metric: str
    reference_policy: str
    candidate_policy: str
    differences_by_seed: dict[int, float]
    statistics: MetricSummary


EpisodeMetric = str | Callable[[EpisodeResult], float]


def summarize_values(values: Iterable[float]) -> MetricSummary:
    """Return mean, sample standard deviation, and a normal 95% CI."""

    samples = np.asarray(list(values), dtype=np.float64)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("At least one scalar value is required")
    if not np.isfinite(samples).all():
        raise ValueError("Summary values must be finite")
    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
    margin = 1.96 * std / float(np.sqrt(samples.size))
    return MetricSummary(
        count=int(samples.size),
        mean=mean,
        std=std,
        ci95_lower=mean - margin,
        ci95_upper=mean + margin,
    )


def summarize_episode_metric(
    results: Iterable[EpisodeResult], metric: EpisodeMetric = "total_reward"
) -> MetricSummary:
    return summarize_values(_metric_value(result, metric) for result in results)


def summarize_paired_seed_differences(
    reference_results: Iterable[EpisodeResult],
    candidate_results: Iterable[EpisodeResult],
    metric: EpisodeMetric = "total_reward",
) -> PairedSeedSummary:
    """Summarize candidate-reference deltas after enforcing exact seed pairing."""

    references = list(reference_results)
    candidates = list(candidate_results)
    reference_by_seed = _results_by_seed(references, "reference")
    candidate_by_seed = _results_by_seed(candidates, "candidate")
    if set(reference_by_seed) != set(candidate_by_seed):
        raise ValueError("Reference and candidate results must contain identical seeds")
    differences = {
        seed: _metric_value(candidate_by_seed[seed], metric)
        - _metric_value(reference_by_seed[seed], metric)
        for seed in sorted(reference_by_seed)
    }
    return PairedSeedSummary(
        metric=_metric_name(metric),
        reference_policy=_single_policy_name(references, "reference"),
        candidate_policy=_single_policy_name(candidates, "candidate"),
        differences_by_seed=differences,
        statistics=summarize_values(differences.values()),
    )


def _metric_value(result: EpisodeResult, metric: EpisodeMetric) -> float:
    if callable(metric):
        return float(metric(result))
    if metric in result.physical_metric_sums:
        return float(result.physical_metric_sums[metric])
    if metric.startswith("physical_metric_sums."):
        return float(result.physical_metric_sums[metric.split(".", 1)[1]])
    if metric.startswith("physical_metric_means."):
        return float(result.physical_metric_means[metric.split(".", 1)[1]])
    value = getattr(result, metric, None)
    if value is None or isinstance(value, (dict, bool)):
        raise ValueError(f"Episode metric is not numeric: {metric}")
    return float(value)


def _metric_name(metric: EpisodeMetric) -> str:
    return metric if isinstance(metric, str) else getattr(metric, "__name__", "custom")


def _results_by_seed(
    results: list[EpisodeResult], label: str
) -> dict[int, EpisodeResult]:
    if not results:
        raise ValueError(f"At least one {label} result is required")
    by_seed = {result.seed: result for result in results}
    if len(by_seed) != len(results):
        raise ValueError(f"Duplicate seeds in {label} results")
    return by_seed


def _single_policy_name(results: list[EpisodeResult], label: str) -> str:
    names = {result.policy for result in results}
    if len(names) != 1:
        raise ValueError(f"Paired {label} results must contain one policy")
    return next(iter(names))


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
        physical_metric_sums = {key: 0.0 for key in PHYSICAL_STEP_METRICS}
        successful_tasks = 0
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(
                policy.act(observation)
            )
            total_reward += float(reward)
            steps += 1
            for key, value in info["reward_components"].items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
            for key in PHYSICAL_STEP_METRICS:
                value = float(info[key])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"Invalid physical step metric {key}: {value}")
                physical_metric_sums[key] += value
            successful_tasks += int(info["reward_components"]["success"] > 0.5)
        physical_metric_means = {
            key: value / steps if steps else 0.0
            for key, value in physical_metric_sums.items()
        }
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
            physical_metric_sums=physical_metric_sums,
            physical_metric_means=physical_metric_means,
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
