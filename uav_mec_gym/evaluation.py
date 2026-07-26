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

AUDIT_STEP_METRICS = (
    "uav_queue_length_total",
    "uav_queue_length_max",
    "active_access_uploads",
    "active_access_downloads",
    "active_backhaul_uploads",
    "active_backhaul_downloads",
    "local_resource_utilization",
    "cloud_resource_utilization",
    "uav_resource_utilization",
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
        # Resource columns 1 and 2 are load and normalized link-delay proxies.
        if mask.size > 0 and mask[0]:
            scores[0] = float(
                observation["resources"][0, 1]
                + observation["resources"][0, 2]
            )
        if mask.size > 1 and mask[1]:
            scores[1] = float(
                observation["resources"][1, 1]
                + observation["resources"][1, 2]
            )
        for uav_id, uav in enumerate(observation["uavs"]):
            target = 2 + uav_id
            if target < mask.size and mask[target]:
                scores[target] = float(uav[4] + uav[6] + uav[7])
        if not np.isfinite(scores).any():
            raise RuntimeError("GymBridge observation has no enabled execution target")
        return {
            "target": int(np.argmin(scores)),
            "movement": np.zeros((self.number_of_uavs, 3), dtype=np.float32),
        }


class _FixedTargetPolicy:
    """Capability baseline that never moves UAVs or changes its target."""

    target: int
    name: str

    def __init__(self, number_of_uavs: int) -> None:
        self.number_of_uavs = number_of_uavs

    def reset(self, seed: int) -> None:
        del seed

    def act(self, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        if self.target >= observation["action_mask"].shape[0]:
            raise RuntimeError("Fixed target is outside the GymBridge action space")
        return {
            "target": self.target,
            "movement": np.zeros((self.number_of_uavs, 3), dtype=np.float32),
        }


class LocalOnlyPolicy(_FixedTargetPolicy):
    name = "local_only"
    target = 0


class CloudOnlyPolicy(_FixedTargetPolicy):
    name = "cloud_only"
    target = 1


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
    audit_metric_means: dict[str, float] = field(default_factory=dict)
    audit_metric_maxima: dict[str, float] = field(default_factory=dict)
    target_counts: dict[str, int] = field(default_factory=dict)
    target_ratios: dict[str, float] = field(default_factory=dict)
    throughput_tasks_per_second: float = 0.0
    deadline_success_rate: float = 0.0
    latency_p95_seconds: float = 0.0
    latency_samples_seconds: list[float] = field(default_factory=list)
    exponential_observation_count: int = 0
    exponential_saturation_count: int = 0
    exponential_saturation_rate: float = 0.0


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
    if metric.startswith("audit_metric_means."):
        return float(result.audit_metric_means[metric.split(".", 1)[1]])
    if metric.startswith("audit_metric_maxima."):
        return float(result.audit_metric_maxima[metric.split(".", 1)[1]])
    if metric.startswith("target_ratios."):
        return float(result.target_ratios[metric.split(".", 1)[1]])
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
        audit_metric_sums = {key: 0.0 for key in AUDIT_STEP_METRICS}
        audit_metric_maxima = {key: 0.0 for key in AUDIT_STEP_METRICS}
        target_counts = {"local": 0, "cloud": 0, "uav": 0}
        successful_tasks = 0
        latency_samples_seconds: list[float] = []
        exponential_observation_count = 0
        exponential_saturation_count = 0
        while not (terminated or truncated):
            exponential = np.asarray(observation["task"][:3], dtype=np.float64)
            if not np.isfinite(exponential).all():
                raise ValueError("Non-finite exponential task observation")
            exponential_observation_count += int(exponential.size)
            exponential_saturation_count += int(np.count_nonzero(exponential >= 1.0))
            action = policy.act(observation)
            target = int(action["target"])
            if target == 0:
                target_counts["local"] += 1
            elif target == 1:
                target_counts["cloud"] += 1
            else:
                target_counts["uav"] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            for key, value in info["reward_components"].items():
                component_sums[key] = component_sums.get(key, 0.0) + float(value)
            for key in PHYSICAL_STEP_METRICS:
                value = float(info[key])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"Invalid physical step metric {key}: {value}")
                physical_metric_sums[key] += value
            for key in AUDIT_STEP_METRICS:
                value = float(info[key])
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"Invalid audit step metric {key}: {value}")
                if key.endswith("_utilization") and value > 1.0:
                    raise ValueError(f"Resource utilization exceeds one: {key}")
                audit_metric_sums[key] += value
                audit_metric_maxima[key] = max(audit_metric_maxima[key], value)
            success_delta = float(info["reward_components"]["success"])
            if success_delta < 0.0 or not np.isclose(success_delta, round(success_delta)):
                raise ValueError("GymBridge success component must be a task count")
            successful_tasks += int(round(success_delta))
            raw_latencies = info.get("settled_task_latencies_seconds")
            if not isinstance(raw_latencies, list):
                raise ValueError(
                    "GymBridge omitted settled_task_latencies_seconds"
                )
            step_latencies = [float(value) for value in raw_latencies]
            if (
                len(step_latencies) != int(info["settled_in_transition"])
                or not np.isfinite(step_latencies).all()
                or any(value < 0.0 for value in step_latencies)
            ):
                raise ValueError("GymBridge returned invalid task latency samples")
            latency_samples_seconds.extend(step_latencies)
        physical_metric_means = {
            key: value / steps if steps else 0.0
            for key, value in physical_metric_sums.items()
        }
        audit_metric_means = {
            key: value / steps if steps else 0.0
            for key, value in audit_metric_sums.items()
        }
        target_ratios = {
            **{
                key: count / steps if steps else 0.0
                for key, count in target_counts.items()
            },
            "offloaded": (
                (target_counts["cloud"] + target_counts["uav"]) / steps
                if steps
                else 0.0
            ),
        }
        simulation_time = float(info["simulation_time"])
        results.append(EpisodeResult(
            policy=policy.name,
            seed=seed,
            steps=steps,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            settled_tasks=int(info["settled_tasks"]),
            total_tasks=int(info["total_tasks"]),
            simulation_time=simulation_time,
            successful_tasks=successful_tasks,
            reward_component_sums=component_sums,
            physical_metric_sums=physical_metric_sums,
            physical_metric_means=physical_metric_means,
            audit_metric_means=audit_metric_means,
            audit_metric_maxima=audit_metric_maxima,
            target_counts=target_counts,
            target_ratios=target_ratios,
            throughput_tasks_per_second=(
                successful_tasks / simulation_time if simulation_time > 0.0 else 0.0
            ),
            deadline_success_rate=(
                successful_tasks / int(info["settled_tasks"])
                if int(info["settled_tasks"]) > 0
                else 0.0
            ),
            latency_p95_seconds=(
                float(np.percentile(latency_samples_seconds, 95.0))
                if latency_samples_seconds
                else 0.0
            ),
            latency_samples_seconds=latency_samples_seconds,
            exponential_observation_count=exponential_observation_count,
            exponential_saturation_count=exponential_saturation_count,
            exponential_saturation_rate=(
                exponential_saturation_count / exponential_observation_count
                if exponential_observation_count
                else 0.0
            ),
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
        LocalOnlyPolicy(args.uavs),
        CloudOnlyPolicy(args.uavs),
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
