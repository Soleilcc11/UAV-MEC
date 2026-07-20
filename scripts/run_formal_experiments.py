#!/usr/bin/env python3
"""Run the resumable protocol-1.1 independent-training-seed experiment matrix."""

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
DEFAULT_CONFIG = REPOSITORY / "experiments/formal_protocol_1_1_10seed.json"


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
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Formal experiments require a clean tracked tree so report Git "
            f"provenance is truthful:\n{status}"
        )


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("format_version", 0)) != 1:
        raise ValueError("Unsupported formal experiment config version")
    if str(config.get("protocol_version")) != "1.1":
        raise ValueError("Formal experiments require GymBridge protocol 1.1")
    budget = int(config.get("interaction_budget", 0))
    if budget < 1024:
        raise ValueError("Formal interaction_budget must be at least 1024")
    algorithms = config.get("algorithms")
    if not isinstance(algorithms, Mapping) or not algorithms:
        raise ValueError("Formal config must contain algorithms")
    all_training_starts: list[int] = []
    for name, algorithm in algorithms.items():
        if algorithm.get("command") not in {"train-ppo", "train-td3"}:
            raise ValueError(f"Unsupported training command for {name}")
        starts = [int(value) for value in algorithm.get("training_seed_starts", [])]
        if len(starts) != 10 or len(set(starts)) != 10:
            raise ValueError(f"{name} must define ten unique training seed starts")
        all_training_starts.extend(starts)
    if len(set(all_training_starts)) != len(all_training_starts):
        raise ValueError("Independent training seed starts overlap across algorithms")
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
    for split in ("training_validation", "heldout"):
        paths = environments.get(split, [])
        if len(paths) != 3:
            raise ValueError(f"{split} must define exactly three XML files")
        for relative in paths:
            if not (REPOSITORY / relative).is_file():
                raise FileNotFoundError(REPOSITORY / relative)


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
    number_of_uavs: int,
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
        str(number_of_uavs),
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
) -> bool:
    if not report_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        return (
            report.get("algorithm") == algorithm
            and int(report.get("interaction_budget", -1)) == interaction_budget
            and int(report.get("starting_training_interactions", -1)) == 0
            and int(report.get("interactions_this_run", -1)) == interaction_budget
            and report.get("checkpoint", {}).get("sha256")
            == _sha256(checkpoint_path)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def _evaluation_is_complete(
    report_path: Path,
    checkpoint_path: Path,
    *,
    split: str,
    seeds: Sequence[int],
) -> bool:
    if not report_path.is_file() or not checkpoint_path.is_file():
        return False
    try:
        report = _read_json(report_path)
        return (
            report.get("split") == split
            and report.get("paired_seeds") == list(seeds)
            and report.get("audit", {}).get("status") == "passed"
            and report.get("checkpoint", {}).get("sha256")
            == _sha256(checkpoint_path)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
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
        *_cli_arguments(run.arguments),
    ]


def _evaluation_command(
    python: Path,
    config: Mapping[str, Any],
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
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--only", choices=["mixed_action_ppo", "mixed_action_td3"])
    parser.add_argument("--from-index", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    _require_clean_tracked_tree()
    python = REPOSITORY / ".venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError(python)
    output_root = REPOSITORY / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    runs = _matrix(config, args.only)
    runs = [run for run in runs if run.index >= args.from_index]
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        raise ValueError("No formal experiment runs selected")

    manifest_path = output_root / "orchestration.json"
    manifest = {
        "format_version": 1,
        "experiment_id": config["experiment_id"],
        "config": str(config_path.relative_to(REPOSITORY)),
        "config_sha256": _sha256(config_path),
        "git_commit_sha": _git_head(),
        "service_lifecycle": "fresh_jvm_per_training_or_evaluation_report",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_runs": [
            {"algorithm": run.algorithm_name, "seed": run.training_seed_start}
            for run in runs
        ],
        "status": "dry_run" if args.dry_run else "running",
    }
    _write_json_atomic(manifest_path, manifest)

    if not args.skip_build:
        _run_logged(
            ["mvn", "clean", "package", "-q", "-DskipTests"],
            output_root / "logs/build.log",
            dry_run=args.dry_run,
        )
    if args.dry_run:
        for run in runs:
            paths = _paths(output_root, run)
            _run_logged(
                _training_command(python, config, run, paths),
                paths["log"],
                dry_run=True,
            )
            for split in ("validation", "heldout"):
                _run_logged(
                    _evaluation_command(python, config, run, paths, split),
                    paths["log"],
                    dry_run=True,
                )
        return

    try:
        for run in runs:
            paths = _paths(output_root, run)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            if not _training_is_complete(
                paths["training"],
                paths["checkpoint"],
                algorithm=run.algorithm_name,
                interaction_budget=int(config["interaction_budget"]),
            ):
                with _gym_bridge(
                    port=int(config["ports"]["training_validation"]),
                    config_paths=config["environments"]["training_validation"],
                    number_of_uavs=int(config["number_of_uavs"]),
                    log_path=paths["directory"] / "gymbridge-training.log",
                ):
                    _run_logged(
                        _training_command(python, config, run, paths),
                        paths["log"],
                        dry_run=False,
                    )
            seeds = config["evaluation_seeds"]["validation"]
            if not _evaluation_is_complete(
                paths["validation"],
                paths["checkpoint"],
                split="validation",
                seeds=seeds,
            ):
                with _gym_bridge(
                    port=int(config["ports"]["training_validation"]),
                    config_paths=config["environments"]["training_validation"],
                    number_of_uavs=int(config["number_of_uavs"]),
                    log_path=paths["directory"] / "gymbridge-validation.log",
                ):
                    _run_logged(
                        _evaluation_command(
                            python, config, run, paths, "validation"
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
                split="heldout",
                seeds=seeds,
            ):
                with _gym_bridge(
                    port=int(config["ports"]["heldout"]),
                    config_paths=config["environments"]["heldout"],
                    number_of_uavs=int(config["number_of_uavs"]),
                    log_path=paths["directory"] / "gymbridge-heldout.log",
                ):
                    _run_logged(
                        _evaluation_command(python, config, run, paths, "heldout"),
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
