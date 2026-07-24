#!/usr/bin/env python3
"""Run the resumable protocol-1.2 independent-training-seed experiment matrix."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "experiments/formal_protocol_1_2_10seed.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _environment_sha256(config_paths: Sequence[str]) -> str:
    entries = [
        {"name": Path(path).name, "sha256": _sha256(REPOSITORY / path)}
        for path in config_paths
    ]
    entries.sort(key=lambda entry: entry["name"])
    return _canonical_hash(entries)


def _repository_source_sha256() -> str:
    files: list[Path] = []
    pom = REPOSITORY / "pom.xml"
    if pom.is_file():
        files.append(pom)
    for relative_root in (Path("src/main/java"), Path("src/main/resources")):
        directory = REPOSITORY / relative_root
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(REPOSITORY).as_posix()):
        digest.update(path.relative_to(REPOSITORY).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_tracked_tree() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Formal experiments require a clean tree so report Git "
            f"provenance is truthful:\n{status}"
        )


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("format_version", 0)) != 2:
        raise ValueError("Unsupported formal experiment config version")
    if str(config.get("protocol_version")) != "1.2":
        raise ValueError("Formal experiments require GymBridge protocol 1.2")
    budget = int(config.get("interaction_budget", 0))
    if budget < 1024:
        raise ValueError("Formal interaction_budget must be at least 1024")
    if int(config.get("bootstrap_resamples", 0)) != 20_000:
        raise ValueError("Formal protocol requires exactly 20,000 bootstrap resamples")
    if config.get("primary_metric") != "deadline_success_rate":
        raise ValueError("Formal primary metric must be deadline_success_rate")
    algorithms = config.get("algorithms")
    if not isinstance(algorithms, Mapping) or not algorithms:
        raise ValueError("Formal config must contain algorithms")
    expected_training_starts: list[int] | None = None
    for name, algorithm in algorithms.items():
        if algorithm.get("command") not in {
            "train-ddpg", "train-ppo", "train-dqn", "train-td3"
        }:
            raise ValueError(f"Unsupported training command for {name}")
        starts = [int(value) for value in algorithm.get("training_seed_starts", [])]
        if len(starts) != 10 or len(set(starts)) != 10:
            raise ValueError(f"{name} must define ten unique training seed starts")
        if expected_training_starts is None:
            expected_training_starts = starts
        elif starts != expected_training_starts:
            raise ValueError(
                "All algorithms must use the same paired training seed starts"
            )
    if (
        "masked_dqn_zero_movement" in algorithms
        and algorithms["masked_dqn_zero_movement"].get("comparison_role")
        != "action_capability_ablation"
    ):
        raise ValueError("DQN must be labeled as an action-capability ablation")
    if any(
        algorithms[name].get("comparison_role") != "main"
        for name in (
            set(algorithms)
            & {
                "masked_parameterized_action_ddpg",
                "mixed_action_ppo",
                "mixed_action_td3",
            }
        )
        if "comparison_role" in algorithms[name]
    ):
        raise ValueError("DDPG, PPO, and TD3 must be labeled as main comparisons")
    if "baselines" in config and config.get("baselines") != [
            "random_masked",
            "minimum_estimated_delay",
            "local_only",
            "cloud_only",
    ]:
        raise ValueError("Formal baseline set or ordering is not frozen")
    if int(config.get("number_of_uavs", 0)) <= 0:
        raise ValueError("Formal protocol requires a positive UAV count")
    evaluation = config.get("evaluation_seeds", {})
    validation = [int(value) for value in evaluation.get("validation", [])]
    heldout = [int(value) for value in evaluation.get("heldout", [])]
    if len(validation) != 10 or len(set(validation)) != 10:
        raise ValueError("validation must contain ten unique paired seeds")
    if len(heldout) != 10 or len(set(heldout)) != 10:
        raise ValueError("heldout must contain ten unique paired seeds")
    if set(validation) & set(heldout):
        raise ValueError("validation and heldout seed sets overlap")
    environments = config.get("environments", {})
    mobile_devices = config.get("mobile_devices", {})
    for split in ("training_validation", "heldout"):
        paths = environments.get(split, [])
        if len(paths) != 3:
            raise ValueError(f"{split} must define exactly three XML files")
        for relative in paths:
            if not (REPOSITORY / relative).is_file():
                raise FileNotFoundError(REPOSITORY / relative)
        if int(mobile_devices.get(split, 0)) <= 0:
            raise ValueError(f"{split} must define a positive mobile device count")


def _cli_arguments(arguments: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in arguments.items():
        option = f"--{key}"
        if isinstance(value, bool):
            if value:
                values.append(option)
        elif isinstance(value, list):
            values.append(option)
            values.extend(str(item) for item in value)
        else:
            values.extend((option, str(value)))
    return values


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_port(process: subprocess.Popen[Any], port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"GymBridge exited during startup with code {process.returncode}"
            )
        if _port_is_open(port):
            return
        time.sleep(0.1)
    raise TimeoutError(f"GymBridge did not listen on port {port} within {timeout}s")


@contextlib.contextmanager
def _gym_bridge(
    *,
    port: int,
    config_paths: Sequence[str],
    number_of_mobile_devices: int,
    log_path: Path,
) -> Iterator[None]:
    if _port_is_open(port):
        raise RuntimeError(f"Refusing to reuse occupied port {port}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "java",
        "-cp",
        "target/classes:target/lib/*",
        "edu.boun.edgecloudsim.uav.GymBridgeMain",
        str(port),
        *(str(REPOSITORY / path) for path in config_paths),
        str(number_of_mobile_devices),
    ]
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_port(process, port, timeout=30.0)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)


def _run_logged(command: Sequence[str], log_path: Path, *, dry_run: bool) -> None:
    print("RUN", " ".join(command), flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        log.flush()
        subprocess.run(
            list(command),
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def _training_is_complete(
    report_path: Path,
    checkpoint_path: Path,
    *,
    algorithm: str,
    interaction_budget: int,
    training_seed_start: int,
    experiment_id: str,
    config_sha256: str,
    git_commit_sha: str,
    environment_sha256: str,
    source_tree_sha256: str,
) -> bool:
    if not report_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        service_runtime = report.get("service_provenance", {}).get("runtime", {})
        formal_experiment = report.get("formal_experiment", {})
        return (
            report.get("algorithm") == algorithm
            and int(report.get("interaction_budget", -1)) == interaction_budget
            and int(report.get("starting_training_interactions", -1)) == 0
            and int(report.get("interactions_this_run", -1)) == interaction_budget
            and report.get("seed_partitions", {}).get("training_seeds_used", [None])[0]
            == training_seed_start
            and formal_experiment.get("experiment_id") == experiment_id
            and formal_experiment.get("config_sha256") == config_sha256
            and report.get("git_commit_sha") == git_commit_sha
            and report.get("environment", {}).get("sha256") == environment_sha256
            and service_runtime.get("git_commit_sha") == git_commit_sha
            and service_runtime.get("source_tree_sha256") == source_tree_sha256
            and service_runtime.get("classes_current") is True
            and report.get("checkpoint", {}).get("sha256")
            == _sha256(checkpoint_path)
        )
    except (
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ):
        return False


def _evaluation_is_complete(
    report_path: Path,
    checkpoint_path: Path,
    *,
    algorithm: str,
    split: str,
    seeds: Sequence[int],
    interaction_budget: int,
    experiment_id: str,
    config_sha256: str,
    git_commit_sha: str,
    environment_sha256: str,
    source_tree_sha256: str,
) -> bool:
    if not report_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        service_runtime = report.get("service_provenance", {}).get("runtime", {})
        formal_experiment = report.get("formal_experiment", {})
        return (
            report.get("split") == split
            and report.get("paired_seeds") == list(seeds)
            and report.get("audit", {}).get("status") == "passed"
            and int(report.get("training_interactions", {}).get(algorithm, -1))
            == interaction_budget
            and formal_experiment.get("experiment_id") == experiment_id
            and formal_experiment.get("config_sha256") == config_sha256
            and report.get("git_commit_sha") == git_commit_sha
            and report.get("environment", {}).get("sha256") == environment_sha256
            and service_runtime.get("git_commit_sha") == git_commit_sha
            and service_runtime.get("source_tree_sha256") == source_tree_sha256
            and service_runtime.get("classes_current") is True
            and report.get("checkpoint", {}).get("sha256")
            == _sha256(checkpoint_path)
        )
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ):
        return False


@dataclass(frozen=True)
class Run:
    algorithm_name: str
    command: str
    evaluation_algorithm: str
    training_seed_start: int
    arguments: Mapping[str, Any]
    index: int


def _matrix(config: Mapping[str, Any], only: str | None) -> list[Run]:
    runs: list[Run] = []
    index = 0
    for name, algorithm in config["algorithms"].items():
        if only is not None and name != only:
            continue
        for seed in algorithm["training_seed_starts"]:
            runs.append(
                Run(
                    algorithm_name=name,
                    command=str(algorithm["command"]),
                    evaluation_algorithm=str(algorithm["evaluation_algorithm"]),
                    training_seed_start=int(seed),
                    arguments=dict(algorithm.get("arguments", {})),
                    index=index,
                )
            )
            index += 1
    return runs


def _paths(output_root: Path, run: Run) -> dict[str, Path]:
    directory = output_root / "runs" / run.algorithm_name / f"seed_{run.training_seed_start}"
    return {
        "directory": directory,
        "checkpoint": directory / "checkpoint.pt",
        "training": directory / "training.json",
        "validation": directory / "validation.json",
        "heldout": directory / "heldout.json",
        "log": directory / "run.log",
    }


def _training_command(
    python: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    run: Run,
    paths: Mapping[str, Path],
) -> list[str]:
    environment = config["environments"]["training_validation"]
    return [
        str(python),
        "-m",
        "uav_mec_gym.experiment",
        run.command,
        "--port",
        str(config["ports"]["training_validation"]),
        "--uavs",
        str(config["number_of_uavs"]),
        "--environment-config",
        *environment,
        "--interaction-budget",
        str(config["interaction_budget"]),
        "--training-seed-start",
        str(run.training_seed_start),
        "--validation-seeds",
        *(str(seed) for seed in config["evaluation_seeds"]["validation"]),
        "--heldout-seeds",
        *(str(seed) for seed in config["evaluation_seeds"]["heldout"]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output",
        str(paths["training"]),
        "--formal-experiment-id",
        str(config["experiment_id"]),
        "--formal-config-sha256",
        config_sha256,
        *_cli_arguments(run.arguments),
    ]


def _evaluation_command(
    python: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    run: Run,
    paths: Mapping[str, Path],
    split: str,
) -> list[str]:
    environment_name = "training_validation" if split == "validation" else "heldout"
    return [
        str(python),
        "-m",
        "uav_mec_gym.experiment",
        "evaluate",
        "--algorithm",
        run.evaluation_algorithm,
        "--port",
        str(config["ports"][environment_name]),
        "--uavs",
        str(config["number_of_uavs"]),
        "--environment-config",
        *config["environments"][environment_name],
        "--split",
        split,
        "--seeds",
        *(str(seed) for seed in config["evaluation_seeds"][split]),
        "--checkpoint",
        str(paths["checkpoint"]),
        "--output",
        str(paths[split]),
        "--formal-experiment-id",
        str(config["experiment_id"]),
        "--formal-config-sha256",
        config_sha256,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--only",
        choices=[
            "masked_parameterized_action_ddpg", "mixed_action_ppo",
            "masked_dqn_zero_movement", "mixed_action_td3",
        ],
    )
    parser.add_argument("--from-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    _require_clean_tracked_tree()
    config_sha256 = _sha256(config_path)
    python = REPOSITORY / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    output_root = REPOSITORY / config["output_root"]
    runs = _matrix(config, args.only)
    runs = [run for run in runs if run.index >= args.from_index]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        raise ValueError("No formal experiment runs selected")

    if args.dry_run:
        if not args.skip_build:
            _run_logged(
                ["mvn", "clean", "package", "-q", "-DskipTests"],
                output_root / "logs/build.log",
                dry_run=True,
            )
        for run in runs:
            paths = _paths(output_root, run)
            _run_logged(
                _training_command(python, config, config_sha256, run, paths),
                paths["log"],
                dry_run=True,
            )
            for split in ("validation", "heldout"):
                _run_logged(
                    _evaluation_command(
                        python, config, config_sha256, run, paths, split
                    ),
                    paths["log"],
                    dry_run=True,
                )
        return

    output_root.mkdir(parents=True, exist_ok=True)
    git_commit_sha = _git_head()
    source_tree_sha256 = _repository_source_sha256()
    environment_sha256 = {
        name: _environment_sha256(paths)
        for name, paths in config["environments"].items()
    }
    manifest_path = output_root / "orchestration.json"
    manifest = {
        "format_version": 2,
        "protocol_version": "1.2",
        "experiment_id": config["experiment_id"],
        "config": str(config_path.relative_to(REPOSITORY)),
        "config_sha256": config_sha256,
        "git_commit_sha": git_commit_sha,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "service_lifecycle": "fresh_jvm_per_training_or_evaluation_report",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_runs": [
            {"algorithm": run.algorithm_name, "seed": run.training_seed_start}
            for run in runs
        ],
        "status": "running",
    }
    _write_json_atomic(manifest_path, manifest)

    if not args.skip_build:
        _run_logged(
            ["mvn", "clean", "package", "-q", "-DskipTests"],
            output_root / "logs/build.log",
            dry_run=False,
        )

    try:
        for run in runs:
            paths = _paths(output_root, run)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            if not _training_is_complete(
                paths["training"],
                paths["checkpoint"],
                algorithm=run.algorithm_name,
                interaction_budget=int(config["interaction_budget"]),
                training_seed_start=run.training_seed_start,
                experiment_id=str(config["experiment_id"]),
                config_sha256=config_sha256,
                git_commit_sha=git_commit_sha,
                environment_sha256=environment_sha256["training_validation"],
                source_tree_sha256=source_tree_sha256,
            ):
                with _gym_bridge(
                    port=int(config["ports"]["training_validation"]),
                    config_paths=config["environments"]["training_validation"],
                    number_of_mobile_devices=int(
                        config["mobile_devices"]["training_validation"]
                    ),
                    log_path=paths["directory"] / "gymbridge-training.log",
                ):
                    _run_logged(
                        _training_command(
                            python, config, config_sha256, run, paths
                        ),
                        paths["log"],
                        dry_run=False,
                    )
            seeds = config["evaluation_seeds"]["validation"]
            if not _evaluation_is_complete(
                paths["validation"],
                paths["checkpoint"],
                algorithm=run.algorithm_name,
                split="validation",
                seeds=seeds,
                interaction_budget=int(config["interaction_budget"]),
                experiment_id=str(config["experiment_id"]),
                config_sha256=config_sha256,
                git_commit_sha=git_commit_sha,
                environment_sha256=environment_sha256["training_validation"],
                source_tree_sha256=source_tree_sha256,
            ):
                with _gym_bridge(
                    port=int(config["ports"]["training_validation"]),
                    config_paths=config["environments"]["training_validation"],
                    number_of_mobile_devices=int(
                        config["mobile_devices"]["training_validation"]
                    ),
                    log_path=paths["directory"] / "gymbridge-validation.log",
                ):
                    _run_logged(
                        _evaluation_command(
                            python, config, config_sha256, run, paths, "validation"
                        ),
                        paths["log"],
                        dry_run=False,
                    )

        for run in runs:
            paths = _paths(output_root, run)
            seeds = config["evaluation_seeds"]["heldout"]
            if not _evaluation_is_complete(
                paths["heldout"],
                paths["checkpoint"],
                algorithm=run.algorithm_name,
                split="heldout",
                seeds=seeds,
                interaction_budget=int(config["interaction_budget"]),
                experiment_id=str(config["experiment_id"]),
                config_sha256=config_sha256,
                git_commit_sha=git_commit_sha,
                environment_sha256=environment_sha256["heldout"],
                source_tree_sha256=source_tree_sha256,
            ):
                with _gym_bridge(
                    port=int(config["ports"]["heldout"]),
                    config_paths=config["environments"]["heldout"],
                    number_of_mobile_devices=int(
                        config["mobile_devices"]["heldout"]
                    ),
                    log_path=paths["directory"] / "gymbridge-heldout.log",
                ):
                    _run_logged(
                        _evaluation_command(
                            python, config, config_sha256, run, paths, "heldout"
                        ),
                        paths["log"],
                        dry_run=False,
                    )
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(manifest_path, manifest)


if __name__ == "__main__":
    main()
