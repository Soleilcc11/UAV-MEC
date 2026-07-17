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
from .contract import UAVMECGymEnv
from .evaluation import (
    PHYSICAL_STEP_METRICS,
    EpisodeResult,
    MinimumEstimatedDelayPolicy,
    RandomMaskedPolicy,
    evaluate_policy,
    summarize_episode_metric,
    summarize_paired_seed_differences,
    summarize_values,
)
from .ppo import MixedActionPPO, PPOConfig


RESULT_FORMAT_VERSION = 2
MIN_PAIRED_SEEDS = 5
SUMMARY_METRICS = (
    "total_reward",
    "steps",
    "settled_tasks",
    "successful_tasks",
    "simulation_time",
    "physical_metric_sums.latency_seconds",
    "physical_metric_sums.ue_energy_joules",
    "physical_metric_sums.uav_energy_joules",
    "physical_metric_sums.constraint_violations",
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
    for row in step_rows:
        for key, value in row["reward_components"].items():
            component_sums[key] = component_sums.get(key, 0.0) + float(value)
        for key, value in row["physical_metrics"].items():
            physical_sums[key] += float(value)
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
        "simulation_time": simulation_time,
        "reward_component_sums": component_sums,
        "physical_metric_sums": physical_sums,
        "physical_metric_means": {
            key: value / len(step_rows) for key, value in physical_sums.items()
        },
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
            sample = agent.sample_action(
                observation, deterministic=False, update_normalizer=True
            )
            next_observation, reward, terminated, truncated, info = env.step(
                sample.action
            )
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
            )
            global_step = int(agent.training_steps)
            if global_step != expected_global_step:
                raise AssertionError("PPO training step counter did not advance once")

            physical = _physical_step_metrics(info)
            components = _reward_components(info)
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


def _policy_configuration(policy: Any) -> dict[str, Any]:
    if isinstance(policy, MixedActionPPO):
        return asdict(policy.config)
    if isinstance(policy, RandomMaskedPolicy):
        return {"target": "uniform_over_action_mask", "movement": "uniform_-1_1"}
    if isinstance(policy, MinimumEstimatedDelayPolicy):
        return {"target": "minimum_observable_transfer_delay", "movement": "zero"}
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
    if not policies or not any(isinstance(policy, MixedActionPPO) for policy in policies):
        raise ValueError("Fair evaluation must include the migrated PPO policy")

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
            int(policy.training_steps) if isinstance(policy, MixedActionPPO) else 0
        )
        for result in results:
            row = asdict(result)
            row.update(
                {
                    "split": split,
                    "algorithm_config": config,
                    "algorithm_config_hash": canonical_hash(config),
                    "environment_config_hash": environment["sha256"],
                    "checkpoint": str(checkpoint) if policy_name == "mixed_action_ppo" else None,
                    "checkpoint_sha256": checkpoint_hash
                    if policy_name == "mixed_action_ppo"
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

    ppo_results = results_by_policy["mixed_action_ppo"]
    paired: dict[str, Any] = {}
    for reference_name, reference_results in results_by_policy.items():
        if reference_name == "mixed_action_ppo":
            continue
        paired[reference_name] = {
            metric: asdict(
                summarize_paired_seed_differences(
                    reference_results, ppo_results, metric
                )
            )
            for metric in SUMMARY_METRICS
        }

    return {
        "format_version": RESULT_FORMAT_VERSION,
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
            if isinstance(policy, MixedActionPPO)
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
            "interpretation": "descriptive smoke evidence; no superiority claim",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and fairly evaluate mixed-action PPO on GymBridge"
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

    evaluate_parser = subparsers.add_parser("evaluate")
    _add_shared_bridge_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--split", choices=["validation", "heldout"], required=True)
    evaluate_parser.add_argument(
        "--seeds", type=int, nargs="+", required=True
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
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
        if args.command == "train-ppo":
            resume_source: dict[str, str] | None = None
            if args.resume_from is not None:
                resume_source = {
                    "path": str(args.resume_from),
                    "sha256": file_sha256(args.resume_from),
                }
                agent = MixedActionPPO.load_checkpoint(args.resume_from)
                if agent.number_of_uavs != args.uavs:
                    raise ValueError("Checkpoint UAV count does not match GymBridge")
                config = agent.config
            else:
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
            starting_training_interactions = int(agent.training_steps)
            episodes, updates = train_ppo(
                env,
                agent,
                interaction_budget=args.interaction_budget,
                training_seed_start=args.training_seed_start,
            )
            used_training_seeds_this_run = [episode["seed"] for episode in episodes]
            if agent.training_seed_start is None or agent.next_training_seed is None:
                raise AssertionError("PPO training seed cursor was not recorded")
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
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "algorithm": agent.name,
                "algorithm_config": config_dict,
                "algorithm_config_hash": canonical_hash(config_dict),
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
            agent = MixedActionPPO.load_checkpoint(args.checkpoint)
            if agent.number_of_uavs != args.uavs:
                raise ValueError("Checkpoint UAV count does not match GymBridge")
            policies = [
                RandomMaskedPolicy(args.uavs),
                MinimumEstimatedDelayPolicy(args.uavs),
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
            _write_json(args.output, report)
    finally:
        env.close()


if __name__ == "__main__":
    main()
