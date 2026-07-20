#!/usr/bin/env python3
"""Audit and summarize the formal independent-training-seed experiment matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "experiments/formal_protocol_1_1_10seed.json"
METRICS = (
    "total_reward",
    "success_rate",
    "latency_per_settled_task_seconds",
    "energy_per_settled_task_joules",
    "constraint_violations",
)
BASELINES = ("random_masked", "minimum_estimated_delay")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_value(row: Mapping[str, Any], metric: str) -> float:
    settled = int(row["settled_tasks"])
    denominator = float(max(settled, 1))
    physical = row["physical_metric_sums"]
    if metric == "total_reward":
        value = float(row["total_reward"])
    elif metric == "success_rate":
        value = float(row["successful_tasks"]) / denominator
    elif metric == "latency_per_settled_task_seconds":
        value = float(physical["latency_seconds"]) / denominator
    elif metric == "energy_per_settled_task_joules":
        value = (
            float(physical["ue_energy_joules"])
            + float(physical["uav_energy_joules"])
        ) / denominator
    elif metric == "constraint_violations":
        value = float(physical["constraint_violations"])
    else:
        raise KeyError(metric)
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {metric}")
    return value


def bootstrap_summary(
    values: Iterable[float], *, seed: int, resamples: int = 20_000
) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires at least two finite values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "bootstrap_95_ci_low": float(low),
        "bootstrap_95_ci_high": float(high),
        "bootstrap_resamples": int(resamples),
        "values": array.tolist(),
    }


def _stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _run_directory(output_root: Path, algorithm: str, seed: int) -> Path:
    return output_root / "runs" / algorithm / f"seed_{seed}"


def _validate_training_report(
    report: Mapping[str, Any],
    *,
    algorithm: str,
    seed: int,
    budget: int,
    checkpoint: Path,
) -> None:
    if report.get("algorithm") != algorithm:
        raise ValueError(f"Training algorithm mismatch for {algorithm}/{seed}")
    if int(report.get("interaction_budget", -1)) != budget:
        raise ValueError(f"Training budget mismatch for {algorithm}/{seed}")
    if int(report.get("starting_training_interactions", -1)) != 0:
        raise ValueError(f"Formal run unexpectedly resumed: {algorithm}/{seed}")
    if int(report.get("interactions_this_run", -1)) != budget:
        raise ValueError(f"Training interaction count mismatch for {algorithm}/{seed}")
    if report["seed_partitions"]["training_seeds_used"][0] != seed:
        raise ValueError(f"Training seed start mismatch for {algorithm}/{seed}")
    transitions = [
        transition
        for episode in report["episodes"]
        for transition in episode["transitions"]
    ]
    if len(transitions) != budget:
        raise ValueError(f"Raw transition count mismatch for {algorithm}/{seed}")
    if [int(row["global_step"]) for row in transitions] != list(
        range(1, budget + 1)
    ):
        raise ValueError(f"Training global-step sequence is not contiguous: {algorithm}/{seed}")
    if report["checkpoint"]["sha256"] != file_sha256(checkpoint):
        raise ValueError(f"Checkpoint digest mismatch for {algorithm}/{seed}")


def _training_curve(
    report: Mapping[str, Any], *, bin_size: int, seed: int
) -> list[dict[str, Any]]:
    bins: dict[int, list[float]] = defaultdict(list)
    for episode in report["episodes"]:
        for transition in episode["transitions"]:
            step = int(transition["global_step"])
            bins[(step - 1) // bin_size].append(float(transition["reward"]))
    return [
        {
            "training_seed_start": seed,
            "interaction_start": index * bin_size + 1,
            "interaction_end": index * bin_size + len(values),
            "mean_step_reward": float(np.mean(values)),
            "transition_count": len(values),
        }
        for index, values in sorted(bins.items())
    ]


def _evaluation_rows(
    report: Mapping[str, Any],
    *,
    algorithm: str,
    training_seed: int,
    split: str,
    expected_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if report.get("audit", {}).get("status") != "passed":
        raise ValueError(f"Evaluation audit failed: {algorithm}/{training_seed}/{split}")
    if report.get("paired_seeds") != list(expected_seeds):
        raise ValueError(f"Paired seeds mismatch: {algorithm}/{training_seed}/{split}")
    candidates: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    for raw in report["episodes"]:
        row = dict(raw)
        if row["policy"] == algorithm:
            row["training_seed_start"] = training_seed
            candidates.append(row)
        elif row["policy"] in BASELINES:
            baselines.append(row)
        else:
            raise ValueError(f"Unexpected policy in evaluation report: {row['policy']}")
    if len(candidates) != len(expected_seeds):
        raise ValueError(f"Candidate episode count mismatch: {algorithm}/{training_seed}/{split}")
    if len(baselines) != len(expected_seeds) * len(BASELINES):
        raise ValueError(f"Baseline episode count mismatch: {algorithm}/{training_seed}/{split}")
    return candidates, baselines


def _baseline_signature(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "policy": row["policy"],
        "seed": row["seed"],
        "steps": row["steps"],
        "total_reward": row["total_reward"],
        "terminated": row["terminated"],
        "truncated": row["truncated"],
        "settled_tasks": row["settled_tasks"],
        "total_tasks": row["total_tasks"],
        "simulation_time": row["simulation_time"],
        "successful_tasks": row["successful_tasks"],
        "reward_component_sums": row["reward_component_sums"],
        "physical_metric_sums": row["physical_metric_sums"],
    }


def _summarize_training_curves(
    curves: Mapping[str, list[list[dict[str, Any]]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for algorithm, replicates in curves.items():
        end_steps = sorted({point["interaction_end"] for curve in replicates for point in curve})
        points = []
        for end_step in end_steps:
            values = [
                point["mean_step_reward"]
                for curve in replicates
                for point in curve
                if point["interaction_end"] == end_step
            ]
            if len(values) != len(replicates):
                raise ValueError(f"Incomplete training curve bin for {algorithm}@{end_step}")
            points.append(
                {
                    "interaction_end": end_step,
                    **bootstrap_summary(
                        values, seed=_stable_seed(algorithm, "training", str(end_step))
                    ),
                }
            )
        result[algorithm] = {"replicates": replicates, "summary": points}
    return result


def summarize(config_path: Path, output: Path | None = None) -> dict[str, Any]:
    config = read_json(config_path)
    output_root = REPOSITORY / config["output_root"]
    orchestration = read_json(output_root / "orchestration.json")
    if orchestration.get("status") != "complete":
        raise ValueError("Formal orchestration is not complete")
    if orchestration.get("experiment_id") != config["experiment_id"]:
        raise ValueError("Orchestration experiment ID does not match config")
    if orchestration.get("config_sha256") != file_sha256(config_path):
        raise ValueError("Orchestration config hash does not match config")
    required_lifecycle = "fresh_jvm_per_training_or_evaluation_report"
    if orchestration.get("service_lifecycle") != required_lifecycle:
        raise ValueError(
            "Formal orchestration did not isolate each report in a fresh JVM"
        )
    budget = int(config["interaction_budget"])
    bin_size = int(config["training_curve_bin_size"])
    training_curves: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    candidate_rows: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_signatures: dict[tuple[str, str, int], dict[str, Any]] = {}
    baseline_rows: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    git_commits: set[str] = set()
    environment_hashes: dict[str, set[str]] = defaultdict(set)
    checkpoint_hashes: dict[str, dict[int, str]] = defaultdict(dict)

    for algorithm, algorithm_config in config["algorithms"].items():
        for seed in algorithm_config["training_seed_starts"]:
            seed = int(seed)
            directory = _run_directory(output_root, algorithm, seed)
            checkpoint = directory / "checkpoint.pt"
            training = read_json(directory / "training.json")
            _validate_training_report(
                training,
                algorithm=algorithm,
                seed=seed,
                budget=budget,
                checkpoint=checkpoint,
            )
            git_commits.add(str(training["git_commit_sha"]))
            environment_hashes["training_validation"].add(
                str(training["environment"]["sha256"])
            )
            checkpoint_hashes[algorithm][seed] = str(training["checkpoint"]["sha256"])
            training_curves[algorithm].append(
                _training_curve(training, bin_size=bin_size, seed=seed)
            )

            for split in ("validation", "heldout"):
                report = read_json(directory / f"{split}.json")
                if report["checkpoint"]["sha256"] != checkpoint_hashes[algorithm][seed]:
                    raise ValueError(f"Evaluation checkpoint mismatch: {algorithm}/{seed}/{split}")
                git_commits.add(str(report["git_commit_sha"]))
                environment_name = "training_validation" if split == "validation" else "heldout"
                environment_hashes[environment_name].add(str(report["environment"]["sha256"]))
                candidates, baselines = _evaluation_rows(
                    report,
                    algorithm=algorithm,
                    training_seed=seed,
                    split=split,
                    expected_seeds=config["evaluation_seeds"][split],
                )
                candidate_rows[algorithm][split].extend(candidates)
                for row in baselines:
                    key = (split, str(row["policy"]), int(row["seed"]))
                    signature = _baseline_signature(row)
                    if key in baseline_signatures and baseline_signatures[key] != signature:
                        raise ValueError(f"Repeated baseline is not deterministic: {key}")
                    baseline_signatures[key] = signature
                    baseline_rows[str(row["policy"])][split][int(row["seed"])] = row

    if len(git_commits) != 1:
        raise ValueError(f"Formal matrix spans multiple Git commits: {sorted(git_commits)}")
    if next(iter(git_commits)) != orchestration.get("git_commit_sha"):
        raise ValueError("Report Git commit does not match orchestration")
    if any(len(hashes) != 1 for hashes in environment_hashes.values()):
        raise ValueError("Formal matrix spans multiple environment hashes per split")

    summaries: dict[str, Any] = {"candidates": {}, "baselines": {}, "paired": {}}
    replicate_means: dict[str, Any] = defaultdict(lambda: defaultdict(dict))
    for algorithm, by_split in candidate_rows.items():
        summaries["candidates"][algorithm] = {}
        summaries["paired"][algorithm] = {}
        for split, rows in by_split.items():
            by_training_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_training_seed[int(row["training_seed_start"])].append(row)
            if len(by_training_seed) != 10:
                raise ValueError(f"{algorithm}/{split} does not have ten training replicates")
            summaries["candidates"][algorithm][split] = {}
            summaries["paired"][algorithm][split] = {}
            for metric in METRICS:
                values = [
                    float(np.mean([metric_value(row, metric) for row in seed_rows]))
                    for _, seed_rows in sorted(by_training_seed.items())
                ]
                replicate_means[algorithm][split][metric] = values
                summaries["candidates"][algorithm][split][metric] = bootstrap_summary(
                    values, seed=_stable_seed(algorithm, split, metric)
                )
                summaries["paired"][algorithm][split][metric] = {}
                for baseline in BASELINES:
                    differences = []
                    reference = baseline_rows[baseline][split]
                    for _, seed_rows in sorted(by_training_seed.items()):
                        differences.append(
                            float(
                                np.mean(
                                    [
                                        metric_value(row, metric)
                                        - metric_value(reference[int(row["seed"])], metric)
                                        for row in seed_rows
                                    ]
                                )
                            )
                        )
                    summaries["paired"][algorithm][split][metric][baseline] = (
                        bootstrap_summary(
                            differences,
                            seed=_stable_seed(
                                algorithm, split, metric, baseline, "paired"
                            ),
                        )
                    )

    for baseline, by_split in baseline_rows.items():
        summaries["baselines"][baseline] = {}
        for split, by_seed in by_split.items():
            summaries["baselines"][baseline][split] = {
                metric: bootstrap_summary(
                    [metric_value(row, metric) for _, row in sorted(by_seed.items())],
                    seed=_stable_seed(baseline, split, metric),
                )
                for metric in METRICS
            }

    payload = {
        "format_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.relative_to(REPOSITORY)),
        "config_sha256": file_sha256(config_path),
        "git_commit_sha": next(iter(git_commits)),
        "interaction_budget_per_training_seed": budget,
        "independent_training_seeds_per_algorithm": 10,
        "paired_evaluation_seeds_per_split": 10,
        "environment_hashes": {
            name: next(iter(hashes)) for name, hashes in environment_hashes.items()
        },
        "checkpoint_hashes": checkpoint_hashes,
        "metrics": list(METRICS),
        "training_curves": _summarize_training_curves(training_curves),
        "replicate_means": replicate_means,
        "summaries": summaries,
        "audit": {
            "status": "passed",
            "checks": [
                "ten independent training seed starts per algorithm",
                f"fixed {budget}-interaction budget per training seed",
                "ten paired validation and held-out environment seeds",
                "fresh JVM for every training and evaluation report",
                "all raw evaluation audits passed",
                "one Git commit and one environment hash per split",
                "checkpoint SHA-256 matches training and evaluation reports",
                "repeated baselines are bitwise-deterministic on reported metrics",
                "uncertainty is bootstrapped across independent training-seed means",
            ],
        },
    }
    destination = output or output_root / "summary/formal_summary.json"
    write_json(destination, payload)
    _write_tables(payload, destination.parent)
    return payload


def _format_interval(summary: Mapping[str, Any], metric: str) -> str:
    scale = 100.0 if metric == "success_rate" else 1.0
    return (
        f"{float(summary['mean']) * scale:.3f} "
        f"[{float(summary['bootstrap_95_ci_low']) * scale:.3f}, "
        f"{float(summary['bootstrap_95_ci_high']) * scale:.3f}]"
    )


def _write_tables(payload: Mapping[str, Any], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / "heldout_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["policy", "metric", "mean", "bootstrap_95_ci_low", "bootstrap_95_ci_high", "n"]
        )
        policies = {
            **payload["summaries"]["baselines"],
            **payload["summaries"]["candidates"],
        }
        for policy, by_split in policies.items():
            for metric in METRICS:
                summary = by_split["heldout"][metric]
                writer.writerow(
                    [
                        policy,
                        metric,
                        summary["mean"],
                        summary["bootstrap_95_ci_low"],
                        summary["bootstrap_95_ci_high"],
                        summary["count"],
                    ]
                )

    labels = {
        "total_reward": "Reward",
        "success_rate": "Success (%)",
        "latency_per_settled_task_seconds": "Latency/task (s)",
        "energy_per_settled_task_joules": "Energy/task (J)",
        "constraint_violations": "Violations",
    }
    policies = {
        **payload["summaries"]["baselines"],
        **payload["summaries"]["candidates"],
    }
    lines = [
        "# Formal held-out results",
        "",
        "Mean and nonparametric bootstrap 95% confidence interval. Learned-policy "
        "intervals use the ten independent training-seed means; baseline intervals "
        "use the ten paired environment seeds.",
        "",
        "| Policy | " + " | ".join(labels[metric] for metric in METRICS) + " |",
        "|---|" + "---:|" * len(METRICS),
    ]
    for policy, by_split in policies.items():
        lines.append(
            f"| {policy} | "
            + " | ".join(
                _format_interval(by_split["heldout"][metric], metric)
                for metric in METRICS
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Audit: **{payload['audit']['status']}**. Git commit: "
            f"`{payload['git_commit_sha']}`.",
        ]
    )
    (directory / "formal_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summarize(args.config.resolve(), args.output.resolve() if args.output else None)


if __name__ == "__main__":
    main()
