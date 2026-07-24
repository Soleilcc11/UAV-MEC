"""Auditable PPO training and paired held-out evaluation on the real GymBridge.

This module never creates simulator transitions itself.  It consumes the frozen
``UAVMECGymEnv`` contract and records the physical metrics returned by Java.
The legacy ``uav_mec_env.py`` is intentionally not imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from .backend import JavaGymBridgeBackend
from .contract import PROTOCOL_VERSION, UAVMECGymEnv
from .evaluation import (
    AUDIT_STEP_METRICS,
    PHYSICAL_STEP_METRICS,
    CloudOnlyPolicy,
    EpisodeResult,
    LocalOnlyPolicy,
    MinimumEstimatedDelayPolicy,
    RandomMaskedPolicy,
    evaluate_policy,
    summarize_episode_metric,
    summarize_paired_seed_differences,
    summarize_values,
)
from .ppo import MixedActionPPO, PPOConfig
from .td3 import MixedActionTD3, TD3Config
from .ddpg import DDPGConfig, MixedActionDDPG
from .dqn import DQNConfig, MaskedDQN


RESULT_FORMAT_VERSION = 3
MIN_PAIRED_SEEDS = 5
SUMMARY_METRICS = (
    "total_reward",
    "steps",
    "settled_tasks",
    "successful_tasks",
    "deadline_success_rate",
    "latency_p95_seconds",
    "throughput_tasks_per_second",
    "simulation_time",
    "physical_metric_sums.latency_seconds",
    "physical_metric_sums.ue_energy_joules",
    "physical_metric_sums.uav_energy_joules",
    "physical_metric_sums.constraint_violations",
    "audit_metric_means.uav_queue_length_total",
    "audit_metric_maxima.uav_queue_length_max",
    "audit_metric_means.local_resource_utilization",
    "audit_metric_means.cloud_resource_utilization",
    "audit_metric_means.uav_resource_utilization",
    "target_ratios.local",
    "target_ratios.cloud",
    "target_ratios.uav",
    "target_ratios.offloaded",
)


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_manifest(paths: Sequence[str | Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("At least one explicit environment config file is required")
    entries: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name in seen_names:
            raise ValueError(f"Duplicate environment config filename: {path.name}")
        seen_names.add(path.name)
        entries.append({"name": path.name, "sha256": file_sha256(path)})
    entries.sort(key=lambda entry: entry["name"])
    return {"files": entries, "sha256": canonical_hash(entries)}


def current_commit_sha(repository: str | Path = ".") -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_source_sha256(repository: str | Path = ".") -> str:
    """Hash the Java build inputs using the same path+NUL+content scheme as Java."""

    root = Path(repository).resolve()
    files: list[Path] = []
    pom = root / "pom.xml"
    if pom.is_file():
        files.append(pom)
    for relative_root in (Path("src/main/java"), Path("src/main/resources")):
        directory = root / relative_root
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _physical_step_metrics(info: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in PHYSICAL_STEP_METRICS:
        if key not in info:
            raise ValueError(f"GymBridge omitted required physical metric: {key}")
        value = float(info[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid physical metric {key}: {value}")
        metrics[key] = value
    return metrics


def _reward_components(info: Mapping[str, Any]) -> dict[str, float]:
    raw = info.get("reward_components")
    if not isinstance(raw, Mapping):
        raise ValueError("GymBridge omitted reward_components")
    components = {key: float(value) for key, value in raw.items()}
    if not components or not all(np.isfinite(value) for value in components.values()):
        raise ValueError("GymBridge returned invalid reward_components")
    return components


def _latency_samples(info: Mapping[str, Any]) -> list[float]:
    raw = info.get("settled_task_latencies_seconds")
    if not isinstance(raw, list):
        raise ValueError("GymBridge omitted settled_task_latencies_seconds")
    values = [float(value) for value in raw]
    if (
        len(values) != int(info.get("settled_in_transition", -1))
        or not np.isfinite(values).all()
        or any(value < 0.0 for value in values)
    ):
        raise ValueError("GymBridge returned invalid task latency samples")
    return values


def _smdp_audit_metrics(info: Mapping[str, Any]) -> dict[str, float | int]:
    nonnegative = (
        "elapsed_simulation_time",
        "throughput_tasks_per_second",
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
    audit: dict[str, float | int] = {}
    for key in nonnegative:
        value = float(info[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid GymBridge audit metric {key}: {value}")
        if key.endswith("_utilization") and value > 1.0:
            raise ValueError(f"GymBridge resource utilization exceeds one: {key}")
        audit[key] = value
    discount = float(info["effective_discount"])
    if not np.isfinite(discount) or not 0.0 <= discount <= 1.0:
        raise ValueError("GymBridge effective_discount must lie in [0, 1]")
    audit["effective_discount"] = discount
    relay = int(info["selected_cloud_relay_uav"])
    if relay < -1:
        raise ValueError("GymBridge returned an invalid cloud relay UAV ID")
    audit["selected_cloud_relay_uav"] = relay
    return audit


def _finalize_training_episode(
    *,
    seed: int,
    step_rows: list[dict[str, Any]],
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    if not step_rows:
        raise ValueError("A training episode must contain at least one transition")
    last = step_rows[-1]
    component_sums: dict[str, float] = {}
    physical_sums = {key: 0.0 for key in PHYSICAL_STEP_METRICS}
    latency_samples: list[float] = []
    for row in step_rows:
        for key, value in row["reward_components"].items():
            component_sums[key] = component_sums.get(key, 0.0) + float(value)
        for key, value in row["physical_metrics"].items():
            physical_sums[key] += float(value)
        latency_samples.extend(row["latency_samples_seconds"])
    exponential_observation_count = sum(
        int(row["exponential_observation_count"]) for row in step_rows
    )
    exponential_saturation_count = sum(
        int(row["exponential_saturation_count"]) for row in step_rows
    )
    settled = int(last["settled_tasks"])
    successful_value = float(component_sums.get("success", 0.0))
    if successful_value < 0.0 or not np.isclose(
        successful_value, round(successful_value)
    ):
        raise ValueError("Training success component must be a task count")
    successful = int(round(successful_value))
    if successful > settled or settled > int(last["total_tasks"]):
        raise ValueError("Training episode task counts are inconsistent")
    simulation_time = float(last["simulation_time"])
    if simulation_time <= 0.0:
        raise ValueError("Terminal simulation_time must preserve the pre-reset clock")
    return {
        "seed": int(seed),
        "steps": len(step_rows),
        "total_reward": float(sum(row["reward"] for row in step_rows)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "settled_tasks": settled,
        "total_tasks": int(last["total_tasks"]),
        "successful_tasks": successful,
        "deadline_success_rate": successful / settled if settled else 0.0,
        "latency_p95_seconds": (
            float(np.percentile(latency_samples, 95.0))
            if latency_samples
            else 0.0
        ),
        "latency_samples_seconds": latency_samples,
        "throughput_tasks_per_second": successful / simulation_time,
        "simulation_time": simulation_time,
        "reward_component_sums": component_sums,
        "physical_metric_sums": physical_sums,
        "physical_metric_means": {
            key: value / len(step_rows) for key, value in physical_sums.items()
        },
        "exponential_observation_count": exponential_observation_count,
        "exponential_saturation_count": exponential_saturation_count,
        "exponential_saturation_rate": (
            exponential_saturation_count / exponential_observation_count
            if exponential_observation_count
            else 0.0
        ),
        "transitions": step_rows,
    }


def train_ppo(
    env: gym.Env,
    agent: MixedActionPPO,
    *,
    interaction_budget: int,
    training_seed_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Train to a cumulative real-environment interaction target.

    ``interaction_budget`` is the desired final value of
    ``agent.training_steps``, not an additional per-invocation allowance.  A
    fresh agent therefore retains the original behaviour, while a restored
    agent performs exactly ``interaction_budget - agent.training_steps`` new
    interactions.  The policy RNG is deliberately not reset at episode
    boundaries; its checkpointed stream continues independently from the
    persisted environment-seed cursor.
    """

    if interaction_budget <= 0:
        raise ValueError("interaction_budget must be positive")
    starting_training_steps = int(agent.training_steps)
    if interaction_budget <= starting_training_steps:
        raise ValueError(
            "interaction_budget is a cumulative target and must exceed the "
            f"checkpoint's {starting_training_steps} training steps"
        )
    episodes: list[dict[str, Any]] = []
    updates: list[dict[str, float]] = []

    while agent.training_steps < interaction_budget:
        seed = agent.reserve_training_episode_seed(training_seed_start)
        observation, reset_info = env.reset(seed=seed)
        if int(reset_info.get("seed", seed)) != seed:
            raise ValueError("GymBridge reset did not echo the requested seed")
        step_rows: list[dict[str, Any]] = []
        episode_terminated = False
        episode_truncated = False

        while not (episode_terminated or episode_truncated):
            exponential = np.asarray(observation["task"][:3], dtype=np.float64)
            if not np.isfinite(exponential).all():
                raise ValueError("Non-finite exponential task observation")
            sample = agent.sample_action(
                observation, deterministic=False, update_normalizer=True
            )
            next_observation, reward, terminated, truncated, info = env.step(
                sample.action
            )
            smdp_audit = _smdp_audit_metrics(info)
            global_step = int(agent.training_steps)
            # store_transition advances the authoritative cumulative counter.
            # Compute the row index and budget boundary from that post-store
            # value below.
            expected_global_step = global_step + 1
            budget_truncated = (
                expected_global_step >= interaction_budget
                and not (terminated or truncated)
            )
            learning_truncated = bool(truncated or budget_truncated)
            agent.store_transition(
                sample,
                float(reward),
                next_observation,
                bool(terminated),
                learning_truncated,
                discount=float(smdp_audit["effective_discount"]),
            )
            global_step = int(agent.training_steps)
            if global_step != expected_global_step:
                raise AssertionError("PPO training step counter did not advance once")

            physical = _physical_step_metrics(info)
            components = _reward_components(info)
            latency_samples = _latency_samples(info)
            step_rows.append(
                {
                    "global_step": global_step,
                    "reward": float(reward),
                    "target": int(sample.action["target"]),
                    "movement": np.asarray(
                        sample.action["movement"], dtype=np.float32
                    ).tolist(),
                    "old_joint_log_prob": float(sample.log_prob),
                    "terminated": bool(terminated),
                    "truncated": learning_truncated,
                    "environment_truncated": bool(truncated),
                    "interaction_budget_truncated": budget_truncated,
                    "settled_tasks": int(info["settled_tasks"]),
                    "total_tasks": int(info["total_tasks"]),
                    "simulation_time": float(info["simulation_time"]),
                    "reward_components": components,
                    "physical_metrics": physical,
                    "latency_samples_seconds": latency_samples,
                    "smdp_audit": smdp_audit,
                    "exponential_observation_count": int(exponential.size),
                    "exponential_saturation_count": int(
                        np.count_nonzero(exponential >= 1.0)
                    ),
                }
            )
            if len(agent.buffer) >= agent.config.rollout_steps:
                update = agent.update()
                if update is not None:
                    updates.append(update)
            observation = next_observation
            episode_terminated = bool(terminated)
            episode_truncated = learning_truncated

        episodes.append(
            _finalize_training_episode(
                seed=seed,
                step_rows=step_rows,
                terminated=episode_terminated,
                truncated=episode_truncated,
            )
        )

    tail_update = agent.update()
    if tail_update is not None:
        updates.append(tail_update)
    if agent.training_steps != interaction_budget:
        raise AssertionError(
            "PPO training step counter diverged from cumulative interaction target"
        )
    return episodes, updates


def train_replay_agent(
    env: gym.Env,
    agent: Any,
    *,
    interaction_budget: int,
    training_seed_start: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    """Train a replay-buffer agent to a cumulative interaction target."""

    if interaction_budget <= 0:
        raise ValueError("interaction_budget must be positive")
    starting_training_steps = int(agent.training_steps)
    if interaction_budget <= starting_training_steps:
        raise ValueError(
            "interaction_budget is a cumulative target and must exceed the "
            f"checkpoint's {starting_training_steps} training steps"
        )
    episodes: list[dict[str, Any]] = []
    updates: list[dict[str, float]] = []

    while agent.training_steps < interaction_budget:
        seed = agent.reserve_training_episode_seed(training_seed_start)
        observation, reset_info = env.reset(seed=seed)
        if int(reset_info.get("seed", seed)) != seed:
            raise ValueError("GymBridge reset did not echo the requested seed")
        step_rows: list[dict[str, Any]] = []
        episode_terminated = False
        episode_truncated = False

        while not (episode_terminated or episode_truncated):
            exponential = np.asarray(observation["task"][:3], dtype=np.float64)
            if not np.isfinite(exponential).all():
                raise ValueError("Non-finite exponential task observation")
            sample = agent.sample_action(
                observation, deterministic=False, update_normalizer=True
            )
            next_observation, reward, terminated, truncated, info = env.step(
                sample.action
            )
            smdp_audit = _smdp_audit_metrics(info)
            expected_global_step = int(agent.training_steps) + 1
            budget_truncated = (
                expected_global_step >= interaction_budget
                and not (terminated or truncated)
            )
            learning_truncated = bool(truncated or budget_truncated)
            agent.store_transition(
                sample,
                float(reward),
                next_observation,
                bool(terminated),
                learning_truncated,
                discount=float(smdp_audit["effective_discount"]),
            )
            global_step = int(agent.training_steps)
            if global_step != expected_global_step:
                raise AssertionError("Replay-agent training step counter did not advance once")
            update = agent.update()
            if update is not None:
                updates.append(update)

            physical = _physical_step_metrics(info)
            components = _reward_components(info)
            latency_samples = _latency_samples(info)
            step_rows.append(
                {
                    "global_step": global_step,
                    "reward": float(reward),
                    "target": int(sample.action["target"]),
                    "movement": np.asarray(
                        sample.action["movement"], dtype=np.float32
                    ).tolist(),
                    "exploration": (
                        {"warmup_random": bool(sample.warmup_random)}
                        if hasattr(sample, "warmup_random")
                        else {"epsilon": float(sample.epsilon)}
                    ),
                    "warmup_random": bool(
                        getattr(sample, "warmup_random", False)
                    ),
                    "terminated": bool(terminated),
                    "truncated": learning_truncated,
                    "environment_truncated": bool(truncated),
                    "interaction_budget_truncated": budget_truncated,
                    "settled_tasks": int(info["settled_tasks"]),
                    "total_tasks": int(info["total_tasks"]),
                    "simulation_time": float(info["simulation_time"]),
                    "reward_components": components,
                    "physical_metrics": physical,
                    "latency_samples_seconds": latency_samples,
                    "smdp_audit": smdp_audit,
                    "exponential_observation_count": int(exponential.size),
                    "exponential_saturation_count": int(
                        np.count_nonzero(exponential >= 1.0)
                    ),
                }
            )
            observation = next_observation
            episode_terminated = bool(terminated)
            episode_truncated = learning_truncated

        episodes.append(
            _finalize_training_episode(
                seed=seed,
                step_rows=step_rows,
                terminated=episode_terminated,
                truncated=episode_truncated,
            )
        )

    if agent.training_steps != interaction_budget:
        raise AssertionError(
            "Replay-agent training step counter diverged from cumulative interaction target"
        )
    return episodes, updates


train_td3 = train_replay_agent


def audit_episode_results(results: Iterable[EpisodeResult]) -> None:
    rows = list(results)
    if not rows:
        raise ValueError("No episode results to audit")
    for result in rows:
        if result.steps <= 0:
            raise ValueError("Evaluation episode contains no transitions")
        if result.terminated == result.truncated:
            raise ValueError("Evaluation episode must end by exactly one terminal mode")
        if not 0 <= result.successful_tasks <= result.settled_tasks <= result.total_tasks:
            raise ValueError("Evaluation task counts are inconsistent")
        if result.simulation_time <= 0.0:
            raise ValueError("Evaluation lost the terminal simulation clock")
        if not np.isclose(
            result.throughput_tasks_per_second,
            result.successful_tasks / result.simulation_time,
        ):
            raise ValueError("Evaluation throughput disagrees with task counts")
        if not np.isclose(
            result.deadline_success_rate,
            result.successful_tasks / max(result.settled_tasks, 1),
        ):
            raise ValueError("Evaluation deadline success rate is inconsistent")
        if len(result.latency_samples_seconds) != result.settled_tasks:
            raise ValueError("Evaluation task latency samples are incomplete")
        if not result.settled_tasks <= result.steps <= result.total_tasks:
            raise ValueError("Arrival decisions and settled task counts are inconsistent")
        if result.terminated and result.steps != result.total_tasks:
            raise ValueError("Natural termination must decide every configured task")
        success_sum = float(result.reward_component_sums.get("success", np.nan))
        if not np.isfinite(success_sum) or not np.isclose(
            success_sum, result.successful_tasks
        ):
            raise ValueError("Success count disagrees with reward components")
        for key in PHYSICAL_STEP_METRICS:
            value = float(result.physical_metric_sums[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Invalid episode physical metric {key}")
        expected_audit_metrics = set(AUDIT_STEP_METRICS)
        if set(result.audit_metric_means) != expected_audit_metrics:
            raise ValueError("Evaluation audit metric means are incomplete")
        if set(result.audit_metric_maxima) != expected_audit_metrics:
            raise ValueError("Evaluation audit metric maxima are incomplete")
        for key in expected_audit_metrics:
            mean = float(result.audit_metric_means[key])
            maximum = float(result.audit_metric_maxima[key])
            if not np.isfinite(mean) or not np.isfinite(maximum):
                raise ValueError(f"Non-finite episode audit metric {key}")
            if mean < 0.0 or maximum < mean:
                raise ValueError(f"Inconsistent episode audit metric {key}")
            if key.endswith("_utilization") and maximum > 1.0:
                raise ValueError(f"Resource utilization exceeds one: {key}")
        if sum(result.target_counts.values()) != result.steps:
            raise ValueError("Target counts do not match decision steps")
        if not np.isclose(
            sum(result.target_ratios[key] for key in ("local", "cloud", "uav")),
            1.0,
        ):
            raise ValueError("Target ratios do not sum to one")
        if not np.isclose(
            result.target_ratios["offloaded"],
            result.target_ratios["cloud"] + result.target_ratios["uav"],
        ):
            raise ValueError("Offload ratio is inconsistent")
        if (
            result.exponential_observation_count != result.steps * 3
            or not 0
            <= result.exponential_saturation_count
            <= result.exponential_observation_count
        ):
            raise ValueError("Exponential observation counts are inconsistent")
        if not np.isclose(
            result.exponential_saturation_rate,
            result.exponential_saturation_count
            / max(result.exponential_observation_count, 1),
        ):
            raise ValueError("Exponential saturation rate is inconsistent")


def _policy_configuration(policy: Any) -> dict[str, Any]:
    if isinstance(policy, (MixedActionPPO, MixedActionDDPG, MaskedDQN, MixedActionTD3)):
        return asdict(policy.config)
    if isinstance(policy, RandomMaskedPolicy):
        return {"target": "uniform_over_action_mask", "movement": "uniform_-1_1"}
    if isinstance(policy, MinimumEstimatedDelayPolicy):
        return {"target": "minimum_observable_transfer_delay", "movement": "zero"}
    if isinstance(policy, LocalOnlyPolicy):
        return {"target": "local_even_if_masked", "movement": "zero"}
    if isinstance(policy, CloudOnlyPolicy):
        return {"target": "cloud_even_if_masked", "movement": "zero"}
    raise TypeError(f"Unsupported policy type: {type(policy).__name__}")


def build_fair_evaluation_report(
    env: gym.Env,
    policies: Sequence[Any],
    *,
    seeds: Sequence[int],
    split: str,
    environment: Mapping[str, Any],
    commit_sha: str,
    checkpoint_path: str | Path,
    service_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unique_seeds = [int(seed) for seed in seeds]
    if len(unique_seeds) < MIN_PAIRED_SEEDS or len(set(unique_seeds)) != len(unique_seeds):
        raise ValueError("Fair evaluation requires at least five unique paired seeds")
    if split not in {"validation", "heldout"}:
        raise ValueError("split must be validation or heldout")
    candidates = [
        policy
        for policy in policies
        if isinstance(policy, (MixedActionPPO, MixedActionDDPG, MaskedDQN, MixedActionTD3))
    ]
    if len(candidates) != 1:
        raise ValueError("Fair evaluation must include exactly one learned candidate")
    candidate = candidates[0]
    candidate_name = candidate.name

    checkpoint = Path(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint)
    results_by_policy: dict[str, list[EpisodeResult]] = {}
    algorithm_configs: dict[str, dict[str, Any]] = {}
    for policy in policies:
        if policy.name in results_by_policy:
            raise ValueError(f"Duplicate policy name: {policy.name}")
        results = evaluate_policy(env, policy, unique_seeds)
        audit_episode_results(results)
        results_by_policy[policy.name] = results
        algorithm_configs[policy.name] = _policy_configuration(policy)

    rows: list[dict[str, Any]] = []
    for policy_name, results in results_by_policy.items():
        config = algorithm_configs[policy_name]
        policy = next(item for item in policies if item.name == policy_name)
        training_interactions = (
            int(policy.training_steps)
        if isinstance(policy, (MixedActionPPO, MixedActionDDPG, MaskedDQN, MixedActionTD3))
            else 0
        )
        for result in results:
            row = asdict(result)
            row.update(
                {
                    "split": split,
                    "algorithm_config": config,
                    "algorithm_config_hash": canonical_hash(config),
                    "environment_config_hash": environment["sha256"],
                    "checkpoint": str(checkpoint) if policy_name == candidate_name else None,
                    "checkpoint_sha256": checkpoint_hash
                    if policy_name == candidate_name
                    else None,
                    "training_interactions": training_interactions,
                    "git_commit_sha": commit_sha,
                }
            )
            rows.append(row)

    summaries: dict[str, Any] = {}
    for policy_name, results in results_by_policy.items():
        policy_summary = {
            metric: asdict(summarize_episode_metric(results, metric))
            for metric in SUMMARY_METRICS
        }
        policy_summary["terminated_rate"] = asdict(
            summarize_values(float(result.terminated) for result in results)
        )
        policy_summary["truncated_rate"] = asdict(
            summarize_values(float(result.truncated) for result in results)
        )
        summaries[policy_name] = policy_summary

    candidate_results = results_by_policy[candidate_name]
    paired: dict[str, Any] = {}
    for reference_name, reference_results in results_by_policy.items():
        if reference_name == candidate_name:
            continue
        paired[reference_name] = {
            metric: asdict(
                summarize_paired_seed_differences(
                    reference_results, candidate_results, metric
                )
            )
            for metric in SUMMARY_METRICS
        }

    return {
        "format_version": RESULT_FORMAT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "paired_seeds": unique_seeds,
        "git_commit_sha": commit_sha,
        "environment": dict(environment),
        "service_provenance": None
        if service_provenance is None
        else dict(service_provenance),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "training_interactions": {
            policy.name: int(policy.training_steps)
            if isinstance(policy, (MixedActionPPO, MixedActionDDPG, MaskedDQN, MixedActionTD3))
            else 0
            for policy in policies
        },
        "algorithm_configs": algorithm_configs,
        "episodes": rows,
        "summaries": summaries,
        "paired_candidate_minus_reference": paired,
        "audit": {
            "status": "passed",
            "checks": [
                "paired seed sets are identical",
                "one action is recorded for each arrived task",
                "concurrent task settlements may aggregate between decision epochs",
                "success <= settled <= total",
                "terminal simulation time is positive",
                "raw physical metrics are finite and non-negative",
            ],
            "interpretation": (
                "formal paired evaluation with uncertainty; claims still require "
                "independent training-seed replication"
                if len(unique_seeds) >= 10 and int(candidate.training_steps) >= 1024
                else "descriptive smoke evidence; no superiority claim"
            ),
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _add_shared_bridge_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--uavs", type=int, required=True)
    parser.add_argument(
        "--environment-config",
        type=Path,
        nargs="+",
        required=True,
        help="Exact XML files used to launch this GymBridge server",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-experiment-id")
    parser.add_argument("--formal-config-sha256")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and fairly evaluate mixed-action RL on GymBridge"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-ppo")
    _add_shared_bridge_arguments(train_parser)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "PPO checkpoint to continue; its algorithm config and RNG state are "
            "authoritative"
        ),
    )
    train_parser.add_argument(
        "--interaction-budget",
        type=int,
        required=True,
        help="Cumulative target for agent.training_steps",
    )
    train_parser.add_argument("--training-seed-start", type=int, required=True)
    train_parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=[201, 202, 203, 204, 205]
    )
    train_parser.add_argument(
        "--heldout-seeds", type=int, nargs="+", default=[301, 302, 303, 304, 305]
    )
    train_parser.add_argument("--rollout-steps", type=int, default=64)
    train_parser.add_argument("--minibatch-size", type=int, default=32)
    train_parser.add_argument("--update-epochs", type=int, default=4)
    train_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64, 64])
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument(
        "--no-bootstrap-truncated", action="store_true"
    )

    ddpg_parser = subparsers.add_parser("train-ddpg")
    _add_shared_bridge_arguments(ddpg_parser)
    ddpg_parser.add_argument("--checkpoint", type=Path, required=True)
    ddpg_parser.add_argument("--resume-from", type=Path)
    ddpg_parser.add_argument("--interaction-budget", type=int, required=True)
    ddpg_parser.add_argument("--training-seed-start", type=int, required=True)
    ddpg_parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=list(range(201, 211))
    )
    ddpg_parser.add_argument(
        "--heldout-seeds", type=int, nargs="+", default=list(range(301, 311))
    )
    ddpg_parser.add_argument("--batch-size", type=int, default=256)
    ddpg_parser.add_argument("--replay-capacity", type=int, default=100_000)
    ddpg_parser.add_argument("--learning-starts", type=int, default=1_000)
    ddpg_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    ddpg_parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    ddpg_parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    ddpg_parser.add_argument("--exploration-noise", type=float, default=0.1)
    ddpg_parser.add_argument("--discrete-exploration", type=float, default=0.1)
    ddpg_parser.add_argument("--target-temperature", type=float, default=1.0)
    ddpg_parser.add_argument("--no-bootstrap-truncated", action="store_true")

    dqn_parser = subparsers.add_parser("train-dqn")
    _add_shared_bridge_arguments(dqn_parser)
    dqn_parser.add_argument("--checkpoint", type=Path, required=True)
    dqn_parser.add_argument("--resume-from", type=Path)
    dqn_parser.add_argument("--interaction-budget", type=int, required=True)
    dqn_parser.add_argument("--training-seed-start", type=int, required=True)
    dqn_parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=list(range(201, 211))
    )
    dqn_parser.add_argument(
        "--heldout-seeds", type=int, nargs="+", default=list(range(301, 311))
    )
    dqn_parser.add_argument("--batch-size", type=int, default=256)
    dqn_parser.add_argument("--replay-capacity", type=int, default=100_000)
    dqn_parser.add_argument("--learning-starts", type=int, default=1_000)
    dqn_parser.add_argument("--target-update-interval", type=int, default=250)
    dqn_parser.add_argument("--epsilon-start", type=float, default=1.0)
    dqn_parser.add_argument("--epsilon-end", type=float, default=0.05)
    dqn_parser.add_argument("--epsilon-decay-steps", type=int, default=10_000)
    dqn_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    dqn_parser.add_argument("--learning-rate", type=float, default=3e-4)
    dqn_parser.add_argument("--no-bootstrap-truncated", action="store_true")

    td3_parser = subparsers.add_parser("train-td3")
    _add_shared_bridge_arguments(td3_parser)
    td3_parser.add_argument("--checkpoint", type=Path, required=True)
    td3_parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "TD3 checkpoint to continue; its algorithm config, replay, and RNG "
            "state are authoritative"
        ),
    )
    td3_parser.add_argument(
        "--interaction-budget",
        type=int,
        required=True,
        help="Cumulative target for agent.training_steps",
    )
    td3_parser.add_argument("--training-seed-start", type=int, required=True)
    td3_parser.add_argument(
        "--validation-seeds", type=int, nargs="+", default=list(range(201, 211))
    )
    td3_parser.add_argument(
        "--heldout-seeds", type=int, nargs="+", default=list(range(301, 311))
    )
    td3_parser.add_argument("--batch-size", type=int, default=256)
    td3_parser.add_argument("--replay-capacity", type=int, default=100_000)
    td3_parser.add_argument("--learning-starts", type=int, default=1_000)
    td3_parser.add_argument("--policy-delay", type=int, default=2)
    td3_parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256])
    td3_parser.add_argument("--actor-learning-rate", type=float, default=3e-4)
    td3_parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    td3_parser.add_argument("--policy-noise", type=float, default=0.2)
    td3_parser.add_argument("--noise-clip", type=float, default=0.5)
    td3_parser.add_argument("--exploration-noise", type=float, default=0.1)
    td3_parser.add_argument("--discrete-exploration", type=float, default=0.1)
    td3_parser.add_argument("--target-temperature", type=float, default=1.0)
    td3_parser.add_argument("--no-bootstrap-truncated", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate")
    _add_shared_bridge_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument(
        "--algorithm", choices=["ddpg", "ppo", "dqn", "td3"], default="ppo"
    )
    evaluate_parser.add_argument("--split", choices=["validation", "heldout"], required=True)
    evaluate_parser.add_argument(
        "--seeds", type=int, nargs="+", required=True
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if (args.formal_experiment_id is None) != (
        args.formal_config_sha256 is None
    ):
        raise ValueError(
            "formal experiment ID and config SHA-256 must be supplied together"
        )
    formal_experiment = None
    if args.formal_experiment_id is not None:
        if len(args.formal_config_sha256) != 64:
            raise ValueError("formal config SHA-256 must contain 64 hex characters")
        try:
            int(args.formal_config_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "formal config SHA-256 must contain 64 hex characters"
            ) from exc
        formal_experiment = {
            "experiment_id": args.formal_experiment_id,
            "config_sha256": args.formal_config_sha256,
        }
    environment = environment_manifest(args.environment_config)
    repository = Path(__file__).resolve().parents[1]
    commit_sha = current_commit_sha(repository)
    source_tree_sha256 = repository_source_sha256(repository)
    backend = JavaGymBridgeBackend(
        host=args.host,
        port=args.port,
        expected_number_of_uavs=args.uavs,
        expected_environment_manifest=environment,
        expected_git_commit_sha=commit_sha,
        expected_source_tree_sha256=source_tree_sha256,
    )
    env = UAVMECGymEnv(args.uavs, backend)
    try:
        if args.command in {"train-ddpg", "train-ppo", "train-dqn", "train-td3"}:
            resume_source: dict[str, str] | None = None
            if args.resume_from is not None:
                resume_source = {
                    "path": str(args.resume_from),
                    "sha256": file_sha256(args.resume_from),
                }
                loaders = {
                    "train-ddpg": MixedActionDDPG.load_checkpoint,
                    "train-ppo": MixedActionPPO.load_checkpoint,
                    "train-dqn": MaskedDQN.load_checkpoint,
                    "train-td3": MixedActionTD3.load_checkpoint,
                }
                agent = loaders[args.command](args.resume_from)
                if agent.number_of_uavs != args.uavs:
                    raise ValueError("Checkpoint UAV count does not match GymBridge")
                config = agent.config
            elif args.command == "train-ppo":
                config = PPOConfig(
                    rollout_steps=args.rollout_steps,
                    minibatch_size=args.minibatch_size,
                    update_epochs=args.update_epochs,
                    hidden_sizes=tuple(args.hidden_sizes),
                    learning_rate=args.learning_rate,
                    bootstrap_truncated=not args.no_bootstrap_truncated,
                )
                agent = MixedActionPPO(
                    args.uavs, config=config, seed=args.training_seed_start
                )
            elif args.command == "train-ddpg":
                config = DDPGConfig(
                    batch_size=args.batch_size,
                    replay_capacity=args.replay_capacity,
                    learning_starts=args.learning_starts,
                    hidden_sizes=tuple(args.hidden_sizes),
                    actor_learning_rate=args.actor_learning_rate,
                    critic_learning_rate=args.critic_learning_rate,
                    exploration_noise=args.exploration_noise,
                    discrete_exploration=args.discrete_exploration,
                    target_temperature=args.target_temperature,
                    bootstrap_truncated=not args.no_bootstrap_truncated,
                )
                agent = MixedActionDDPG(
                    args.uavs, config=config, seed=args.training_seed_start
                )
            elif args.command == "train-dqn":
                config = DQNConfig(
                    batch_size=args.batch_size,
                    replay_capacity=args.replay_capacity,
                    learning_starts=args.learning_starts,
                    target_update_interval=args.target_update_interval,
                    epsilon_start=args.epsilon_start,
                    epsilon_end=args.epsilon_end,
                    epsilon_decay_steps=args.epsilon_decay_steps,
                    hidden_sizes=tuple(args.hidden_sizes),
                    learning_rate=args.learning_rate,
                    bootstrap_truncated=not args.no_bootstrap_truncated,
                )
                agent = MaskedDQN(
                    args.uavs, config=config, seed=args.training_seed_start
                )
            else:
                config = TD3Config(
                    batch_size=args.batch_size,
                    replay_capacity=args.replay_capacity,
                    learning_starts=args.learning_starts,
                    policy_delay=args.policy_delay,
                    hidden_sizes=tuple(args.hidden_sizes),
                    actor_learning_rate=args.actor_learning_rate,
                    critic_learning_rate=args.critic_learning_rate,
                    policy_noise=args.policy_noise,
                    noise_clip=args.noise_clip,
                    exploration_noise=args.exploration_noise,
                    discrete_exploration=args.discrete_exploration,
                    target_temperature=args.target_temperature,
                    bootstrap_truncated=not args.no_bootstrap_truncated,
                )
                agent = MixedActionTD3(
                    args.uavs, config=config, seed=args.training_seed_start
                )
            starting_training_interactions = int(agent.training_steps)
            training_function = train_ppo if args.command == "train-ppo" else train_replay_agent
            episodes, updates = training_function(
                env,
                agent,
                interaction_budget=args.interaction_budget,
                training_seed_start=args.training_seed_start,
            )
            used_training_seeds_this_run = [episode["seed"] for episode in episodes]
            if agent.training_seed_start is None or agent.next_training_seed is None:
                raise AssertionError("training seed cursor was not recorded")
            used_training_seeds = list(
                range(agent.training_seed_start, agent.next_training_seed)
            )
            seed_partitions = {
                "training": set(used_training_seeds),
                "validation": set(args.validation_seeds),
                "heldout": set(args.heldout_seeds),
            }
            if any(
                seed_partitions[left] & seed_partitions[right]
                for left, right in (
                    ("training", "validation"),
                    ("training", "heldout"),
                    ("validation", "heldout"),
                )
            ):
                raise ValueError("Training, validation, and held-out seed partitions overlap")
            agent.save_checkpoint(args.checkpoint)
            checkpoint_hash = file_sha256(args.checkpoint)
            config_dict = asdict(config)
            payload = {
                "format_version": RESULT_FORMAT_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "algorithm": agent.name,
                "algorithm_config": config_dict,
                "algorithm_config_hash": canonical_hash(config_dict),
                "formal_experiment": formal_experiment,
                "environment": environment,
                "service_provenance": backend.provenance,
                "git_commit_sha": commit_sha,
                "interaction_budget": args.interaction_budget,
                "interaction_budget_semantics": "cumulative_training_steps_target",
                "starting_training_interactions": starting_training_interactions,
                "interactions_this_run": (
                    args.interaction_budget - starting_training_interactions
                ),
                "resume_source": resume_source,
                "seed_partitions": {
                    "training_seeds_used": used_training_seeds,
                    "training_seeds_used_this_run": used_training_seeds_this_run,
                    "next_training_seed": agent.next_training_seed,
                    "validation_seeds_reserved": list(args.validation_seeds),
                    "heldout_seeds_reserved": list(args.heldout_seeds),
                },
                "checkpoint": {
                    "path": str(args.checkpoint),
                    "sha256": checkpoint_hash,
                    "selection": "final cumulative-interaction-target checkpoint",
                },
                "episodes": episodes,
                "updates": updates,
            }
            _write_json(args.output, payload)
        else:
            loaders = {
                "ddpg": MixedActionDDPG.load_checkpoint,
                "ppo": MixedActionPPO.load_checkpoint,
                "dqn": MaskedDQN.load_checkpoint,
                "td3": MixedActionTD3.load_checkpoint,
            }
            agent = loaders[args.algorithm](args.checkpoint)
            if agent.number_of_uavs != args.uavs:
                raise ValueError("Checkpoint UAV count does not match GymBridge")
            policies = [
                RandomMaskedPolicy(args.uavs),
                MinimumEstimatedDelayPolicy(args.uavs),
                LocalOnlyPolicy(args.uavs),
                CloudOnlyPolicy(args.uavs),
                agent,
            ]
            report = build_fair_evaluation_report(
                env,
                policies,
                seeds=args.seeds,
                split=args.split,
                environment=environment,
                commit_sha=commit_sha,
                checkpoint_path=args.checkpoint,
                service_provenance=backend.provenance,
            )
            report["formal_experiment"] = formal_experiment
            _write_json(args.output, report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
