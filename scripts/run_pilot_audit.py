#!/usr/bin/env python3
"""Run and audit the one-seed GymBridge protocol-1.2 pilot matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from scripts.run_formal_experiments import (
        REPOSITORY,
        _cli_arguments,
        _gym_bridge,
        _read_json,
        _require_clean_tracked_tree,
        _run_logged,
        _sha256,
        _write_json_atomic,
    )
except ModuleNotFoundError:
    from run_formal_experiments import (
        REPOSITORY,
        _cli_arguments,
        _gym_bridge,
        _read_json,
        _require_clean_tracked_tree,
        _run_logged,
        _sha256,
        _write_json_atomic,
    )


DEFAULT_CONFIG = REPOSITORY / "experiments/pilot_protocol_1_2_1seed.json"
EXPECTED_ALGORITHMS = {
    "masked_parameterized_action_ddpg",
    "mixed_action_ppo",
    "mixed_action_td3",
    "masked_dqn_zero_movement",
}


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("format_version", 0)) != 1:
        raise ValueError("Unsupported pilot config version")
    if config.get("protocol_version") != "1.2":
        raise ValueError("Pilot requires GymBridge protocol 1.2")
    if int(config.get("number_of_uavs", 0)) != 1:
        raise ValueError("Pilot requires exactly one UAV")
    if int(config.get("mobile_devices", 0)) != 20:
        raise ValueError("Pilot requires exactly 20 mobile devices")
    if int(config.get("interaction_budget", 0)) != 512:
        raise ValueError("Pilot requires exactly 512 interactions per algorithm")
    if set(config.get("algorithms", {})) != EXPECTED_ALGORITHMS:
        raise ValueError("Pilot must cover all four frozen algorithms")
    environment = list(config.get("environment", []))
    if len(environment) != 3:
        raise ValueError("Pilot must define exactly three environment XML files")
    for relative in environment:
        if not (REPOSITORY / relative).is_file():
            raise FileNotFoundError(REPOSITORY / relative)
    evaluation = [int(value) for value in config.get("evaluation_seeds", [])]
    heldout = [int(value) for value in config.get("heldout_seeds_reserved", [])]
    if len(evaluation) < 5 or len(evaluation) != len(set(evaluation)):
        raise ValueError("Pilot requires at least five unique evaluation seeds")
    if len(heldout) < 5 or len(heldout) != len(set(heldout)):
        raise ValueError("Pilot requires at least five unique reserved held-out seeds")
    training_seed = int(config.get("training_seed_start", -1))
    if training_seed < 0 or training_seed in set(evaluation) | set(heldout):
        raise ValueError("Pilot seed partitions overlap")
    limit = float(config.get("exponential_saturation_rate_limit", -1.0))
    if not 0.0 < limit <= 1.0:
        raise ValueError("Pilot saturation limit must lie in (0, 1]")


def _paths(output_root: Path, algorithm: str) -> dict[str, Path]:
    directory = output_root / algorithm
    return {
        "directory": directory,
        "checkpoint": directory / "checkpoint.pt",
        "training": directory / "training.json",
        "evaluation": directory / "evaluation.json",
        "log": directory / "run.log",
    }


def _training_command(
    python: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    algorithm: str,
    paths: Mapping[str, Path],
) -> list[str]:
    specification = config["algorithms"][algorithm]
    return [
        str(python),
        "-m",
        "uav_mec_gym.experiment",
        str(specification["command"]),
        "--port",
        str(config["port"]),
        "--uavs",
        str(config["number_of_uavs"]),
        "--environment-config",
        *[str(path) for path in config["environment"]],
        "--interaction-budget",
        str(config["interaction_budget"]),
        "--training-seed-start",
        str(config["training_seed_start"]),
        "--validation-seeds",
        *[str(seed) for seed in config["evaluation_seeds"]],
        "--heldout-seeds",
        *[str(seed) for seed in config["heldout_seeds_reserved"]],
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output",
        str(paths["training"]),
        "--formal-experiment-id",
        str(config["experiment_id"]),
        "--formal-config-sha256",
        config_sha256,
        *_cli_arguments(specification.get("arguments", {})),
    ]


def _evaluation_command(
    python: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    algorithm: str,
    paths: Mapping[str, Path],
) -> list[str]:
    return [
        str(python),
        "-m",
        "uav_mec_gym.experiment",
        "evaluate",
        "--algorithm",
        str(config["algorithms"][algorithm]["evaluation_algorithm"]),
        "--port",
        str(config["port"]),
        "--uavs",
        str(config["number_of_uavs"]),
        "--environment-config",
        *[str(path) for path in config["environment"]],
        "--split",
        "validation",
        "--seeds",
        *[str(seed) for seed in config["evaluation_seeds"]],
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output",
        str(paths["evaluation"]),
        "--formal-experiment-id",
        str(config["experiment_id"]),
        "--formal-config-sha256",
        config_sha256,
    ]


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Non-finite value at {path}")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _audit_episode(episode: Mapping[str, Any], label: str) -> tuple[int, int]:
    steps = int(episode["steps"])
    settled = int(episode["settled_tasks"])
    total = int(episode["total_tasks"])
    successful = int(episode["successful_tasks"])
    simulation_time = float(episode["simulation_time"])
    throughput = float(episode["throughput_tasks_per_second"])
    if steps <= 0 or simulation_time <= 0.0:
        raise ValueError(f"{label} lost decisions or simulation clock")
    if not 0 <= successful <= settled <= total:
        raise ValueError(f"{label} has inconsistent task counts")
    if not math.isclose(
        throughput, successful / simulation_time, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError(f"{label} throughput disagrees with task count and clock")
    if len(episode["latency_samples_seconds"]) != settled:
        raise ValueError(f"{label} latency samples do not match settled tasks")
    observed = int(episode["exponential_observation_count"])
    saturated = int(episode["exponential_saturation_count"])
    if observed != steps * 3 or not 0 <= saturated <= observed:
        raise ValueError(f"{label} exponential observation audit is inconsistent")
    return observed, saturated


def audit(
    config: Mapping[str, Any],
    config_sha256: str,
    output_root: Path,
) -> dict[str, Any]:
    observed = 0
    saturated = 0
    git_commits: set[str] = set()
    source_hashes: set[str] = set()
    environment_hashes: set[str] = set()
    checkpoints: dict[str, str] = {}
    report_paths: dict[str, dict[str, str]] = {}

    for algorithm in config["algorithms"]:
        paths = _paths(output_root, algorithm)
        training = _read_json(paths["training"])
        evaluation = _read_json(paths["evaluation"])
        _assert_finite(training, f"{algorithm}.training")
        _assert_finite(evaluation, f"{algorithm}.evaluation")
        if training.get("algorithm") != algorithm:
            raise ValueError(f"Training algorithm mismatch for {algorithm}")
        if int(training.get("interactions_this_run", -1)) != int(
            config["interaction_budget"]
        ):
            raise ValueError(f"Training budget mismatch for {algorithm}")
        if evaluation.get("audit", {}).get("status") != "passed":
            raise ValueError(f"Evaluation audit failed for {algorithm}")
        if evaluation.get("paired_seeds") != config["evaluation_seeds"]:
            raise ValueError(f"Evaluation seeds mismatch for {algorithm}")
        checkpoint_hash = _sha256(paths["checkpoint"])
        if (
            training["checkpoint"]["sha256"] != checkpoint_hash
            or evaluation["checkpoint"]["sha256"] != checkpoint_hash
        ):
            raise ValueError(f"Checkpoint hash mismatch for {algorithm}")
        checkpoints[algorithm] = checkpoint_hash
        for episode_index, episode in enumerate(training["episodes"]):
            episode_observed, episode_saturated = _audit_episode(
                episode, f"{algorithm}.training[{episode_index}]"
            )
            observed += episode_observed
            saturated += episode_saturated
        for episode_index, episode in enumerate(evaluation["episodes"]):
            episode_observed, episode_saturated = _audit_episode(
                episode, f"{algorithm}.evaluation[{episode_index}]"
            )
            observed += episode_observed
            saturated += episode_saturated
        for report in (training, evaluation):
            if report.get("formal_experiment") != {
                "experiment_id": config["experiment_id"],
                "config_sha256": config_sha256,
            }:
                raise ValueError(f"Pilot config provenance mismatch for {algorithm}")
            git_commits.add(str(report["git_commit_sha"]))
            runtime = report["service_provenance"]["runtime"]
            if runtime.get("classes_current") is not True:
                raise ValueError(f"Stale Java classes reported for {algorithm}")
            if runtime["git_commit_sha"] != report["git_commit_sha"]:
                raise ValueError(f"Java/Python Git provenance mismatch for {algorithm}")
            source_hashes.add(str(runtime["source_tree_sha256"]))
            environment_hashes.add(str(report["environment"]["sha256"]))
        report_paths[algorithm] = {
            "training": str(paths["training"]),
            "evaluation": str(paths["evaluation"]),
            "checkpoint": str(paths["checkpoint"]),
        }

    saturation_rate = saturated / observed if observed else 0.0
    limit = float(config["exponential_saturation_rate_limit"])
    if saturation_rate > limit:
        raise ValueError(
            f"Exponential saturation rate {saturation_rate:.6f} exceeds {limit:.6f}"
        )
    if len(git_commits) != 1 or len(source_hashes) != 1 or len(environment_hashes) != 1:
        raise ValueError("Pilot reports do not share one source/environment provenance")
    return {
        "format_version": 1,
        "protocol_version": "1.2",
        "experiment_id": config["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "git_commit_sha": next(iter(git_commits)),
        "source_tree_sha256": next(iter(source_hashes)),
        "environment_sha256": next(iter(environment_hashes)),
        "config_sha256": config_sha256,
        "checkpoint_sha256": checkpoints,
        "exponential_observations": observed,
        "exponential_saturations": saturated,
        "exponential_saturation_rate": saturation_rate,
        "exponential_saturation_rate_limit": limit,
        "reports": report_paths,
        "checks": [
            "four algorithms completed the same 512-interaction budget",
            "all reports contain only finite numeric values",
            "task counts, throughput, latency samples, and simulation clocks agree",
            "exponential observation saturation stayed within the frozen limit",
            "Git, Java source, environment, config, and checkpoint hashes agree",
            "every training and evaluation report used a fresh JVM",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    config_sha256 = _sha256(config_path)
    python = REPOSITORY / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    output_root = REPOSITORY / config["output_root"]

    if not args.dry_run:
        _require_clean_tracked_tree()
        output_root.mkdir(parents=True, exist_ok=True)
        if not args.skip_build:
            subprocess.run(
                ["mvn", "clean", "package", "-q", "-DskipTests"],
                cwd=REPOSITORY,
                check=True,
            )

    for algorithm in config["algorithms"]:
        paths = _paths(output_root, algorithm)
        if not args.dry_run:
            paths["directory"].mkdir(parents=True, exist_ok=True)
        training_command = _training_command(
            python, config, config_sha256, algorithm, paths
        )
        evaluation_command = _evaluation_command(
            python, config, config_sha256, algorithm, paths
        )
        if args.dry_run:
            _run_logged(training_command, paths["log"], dry_run=True)
            _run_logged(evaluation_command, paths["log"], dry_run=True)
            continue
        for command in (training_command, evaluation_command):
            with _gym_bridge(
                port=int(config["port"]),
                config_paths=config["environment"],
                number_of_mobile_devices=int(config["mobile_devices"]),
                log_path=paths["log"],
            ):
                _run_logged(command, paths["log"], dry_run=False)

    if args.dry_run:
        return
    result = audit(config, config_sha256, output_root)
    _write_json_atomic(output_root / "pilot_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
