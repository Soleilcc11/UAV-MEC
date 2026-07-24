#!/usr/bin/env python3
"""Create publication-ready vector and 300-DPI figures from audited results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = REPOSITORY / "experiments/formal_protocol_1_2_10seed.json"
DEFAULT_SUMMARY = (
    REPOSITORY
    / json.loads(FORMAL_CONFIG.read_text(encoding="utf-8"))["output_root"]
    / "summary/formal_summary.json"
)
ALGORITHM_LABELS = {
    "masked_parameterized_action_ddpg": "Masked parameterized-action DDPG",
    "mixed_action_ppo": "Mixed-action PPO",
    "masked_dqn_zero_movement": "Masked DQN (zero-movement ablation)",
    "mixed_action_td3": "Mixed-action TD3",
    "minimum_estimated_delay": "Min. estimated delay",
    "random_masked": "Masked random",
    "local_only": "Local only",
    "cloud_only": "Cloud only",
}
COLORS = {
    "masked_parameterized_action_ddpg": "#009E73",
    "mixed_action_ppo": "#0072B2",
    "mixed_action_td3": "#D55E00",
    "masked_dqn_zero_movement": "#CC79A7",
    "minimum_estimated_delay": "#222222",
    "random_masked": "#777777",
    "local_only": "#56B4E9",
    "cloud_only": "#E69F00",
}
MARKERS = {
    "masked_parameterized_action_ddpg": "P",
    "mixed_action_ppo": "o",
    "mixed_action_td3": "s",
    "masked_dqn_zero_movement": "X",
    "minimum_estimated_delay": "D",
    "random_masked": "^",
    "local_only": "v",
    "cloud_only": ">",
}
LEARNED_ALGORITHMS = (
    "masked_parameterized_action_ddpg",
    "mixed_action_ppo",
    "masked_dqn_zero_movement",
    "mixed_action_td3",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("audit", {}).get("status") != "passed":
        raise ValueError("Refusing to plot a summary whose audit did not pass")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.titleweight": "normal",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.28,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "svg.fonttype": "none",
            "svg.hashsalt": "uav-mec-formal-protocol-1-2",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#888888")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(
    figure: plt.Figure, output_dir: Path, stem: str, summary_sha256: str
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    metadata = {
        "Title": stem.replace("_", " ").title(),
        "Subject": f"UAV-MEC formal experiment; source SHA-256 {summary_sha256}",
        "Creator": "UAV-MEC reproducible plotting pipeline",
    }
    for suffix in ("pdf", "svg", "png"):
        path = output_dir / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.03}
        if suffix == "png":
            kwargs["dpi"] = 300
        elif suffix == "pdf":
            kwargs["metadata"] = metadata
        figure.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(figure)
    return outputs


def plot_training_curves(
    summary: Mapping[str, Any], output_dir: Path, summary_sha256: str
) -> list[str]:
    figure, axis = plt.subplots(figsize=(7.15, 3.25), constrained_layout=True)
    for algorithm in LEARNED_ALGORITHMS:
        if algorithm not in summary["training_curves"]:
            continue
        points = summary["training_curves"][algorithm]["summary"]
        x = np.asarray([point["interaction_end"] for point in points], dtype=float)
        mean = np.asarray([point["mean"] for point in points], dtype=float)
        low = np.asarray(
            [point["bootstrap_95_ci_low"] for point in points], dtype=float
        )
        high = np.asarray(
            [point["bootstrap_95_ci_high"] for point in points], dtype=float
        )
        axis.fill_between(x, low, high, color=COLORS[algorithm], alpha=0.16, linewidth=0)
        axis.plot(
            x,
            mean,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            markevery=max(1, len(x) // 8),
            label=ALGORITHM_LABELS[algorithm],
        )
    axis.set_xlabel("Real GymBridge interactions")
    axis.set_ylabel("Mean reward per interaction")
    axis.set_title("Training reward across 10 independent seeds", loc="left")
    axis.legend(frameon=False, ncol=2, loc="best")
    _finish_axis(axis)
    return _save_figure(figure, output_dir, "training_curves", summary_sha256)


def _policy_values(
    summary: Mapping[str, Any], policy: str, metric: str
) -> tuple[np.ndarray, Mapping[str, Any]]:
    if policy in summary["summaries"]["candidates"]:
        values = summary["replicate_means"][policy]["heldout"][metric]
        stats = summary["summaries"]["candidates"][policy]["heldout"][metric]
    else:
        stats = summary["summaries"]["baselines"][policy]["heldout"][metric]
        values = stats["values"]
    return np.asarray(values, dtype=float), stats


def plot_heldout_performance(
    summary: Mapping[str, Any], output_dir: Path, summary_sha256: str
) -> list[str]:
    available_metrics = (
        ("total_reward", "Episode reward", 1.0),
        ("deadline_success_rate", "Deadline success (%)", 100.0),
        ("latency_per_settled_task_seconds", "Latency per settled task (s)", 1.0),
        ("latency_p95_seconds", "P95 task latency (s)", 1.0),
        ("energy_per_settled_task_joules", "Energy per settled task (J)", 1.0),
        ("throughput_tasks_per_second", "Throughput (tasks/s)", 1.0),
        ("constraint_violations", "Constraint violations", 1.0),
        ("average_uav_queue_length", "Mean UAV queue length", 1.0),
        ("max_uav_queue_length", "Maximum UAV queue length", 1.0),
        ("uav_resource_utilization", "Mean UAV utilization (%)", 100.0),
        ("offload_ratio", "Offloaded decisions (%)", 100.0),
    )
    baseline_metrics = summary["summaries"]["baselines"][
        "minimum_estimated_delay"
    ]["heldout"]
    metrics = tuple(item for item in available_metrics if item[0] in baseline_metrics)
    policies = (
        "minimum_estimated_delay",
        "random_masked",
        "local_only",
        "cloud_only",
        *(name for name in LEARNED_ALGORITHMS
          if name in summary["summaries"]["candidates"]),
    )
    row_count = (len(metrics) + 1) // 2
    figure, axes = plt.subplots(
        row_count, 2, figsize=(7.15, 2.45 * row_count),
        constrained_layout=True, squeeze=False,
    )
    rng = np.random.default_rng(20260718)
    for axis, (metric, ylabel, scale) in zip(axes.flat, metrics):
        for index, policy in enumerate(policies):
            values, stats = _policy_values(summary, policy, metric)
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            axis.scatter(
                np.full(values.size, index) + jitter,
                values * scale,
                s=14,
                marker=MARKERS[policy],
                facecolors="none",
                edgecolors=COLORS[policy],
                linewidths=0.8,
                alpha=0.72,
                zorder=2,
            )
            mean = float(stats["mean"]) * scale
            low = float(stats["bootstrap_95_ci_low"]) * scale
            high = float(stats["bootstrap_95_ci_high"]) * scale
            axis.errorbar(
                index,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt=MARKERS[policy],
                color=COLORS[policy],
                markerfacecolor=COLORS[policy],
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.4,
                elinewidth=1.2,
                zorder=3,
            )
        axis.set_xticks(range(len(policies)))
        axis.set_xticklabels(
            [ALGORITHM_LABELS[policy] for policy in policies], rotation=18, ha="right"
        )
        axis.set_ylabel(ylabel)
        _finish_axis(axis)
    for axis in axes.flat[len(metrics):]:
        axis.set_visible(False)
    axes[0, 0].set_title("Held-out performance: points and bootstrap 95% CI", loc="left")
    return _save_figure(
        figure, output_dir, "heldout_performance", summary_sha256
    )


def plot_paired_effects(
    summary: Mapping[str, Any], output_dir: Path, summary_sha256: str
) -> list[str]:
    metrics = (
        ("total_reward", "Reward difference"),
        (
            "deadline_success_rate",
            "Deadline-success difference (percentage points)",
        ),
        ("latency_per_settled_task_seconds", "Latency/task difference (s)"),
        ("energy_per_settled_task_joules", "Energy/task difference (J)"),
    )
    algorithms = tuple(
        name for name in LEARNED_ALGORITHMS
        if name in summary["summaries"]["paired"]
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.65), constrained_layout=True)
    for axis, (metric, xlabel) in zip(axes.flat, metrics, strict=True):
        scale = 100.0 if metric == "deadline_success_rate" else 1.0
        axis.axvline(0.0, color="#555555", linewidth=0.8, zorder=0)
        for index, algorithm in enumerate(algorithms):
            stats = summary["summaries"]["paired"][algorithm]["heldout"][metric][
                "minimum_estimated_delay"
            ]
            mean = float(stats["mean"]) * scale
            low = float(stats["bootstrap_95_ci_low"]) * scale
            high = float(stats["bootstrap_95_ci_high"]) * scale
            axis.errorbar(
                mean,
                index,
                xerr=[[mean - low], [high - mean]],
                fmt=MARKERS[algorithm],
                color=COLORS[algorithm],
                markerfacecolor=COLORS[algorithm],
                markeredgecolor="white",
                capsize=2.5,
                elinewidth=1.3,
            )
        axis.set_yticks(range(len(algorithms)))
        axis.set_yticklabels([ALGORITHM_LABELS[name] for name in algorithms])
        axis.set_xlabel(xlabel)
        axis.grid(axis="x", color="#888888")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 0].set_title("Candidate minus minimum-delay baseline (held-out)", loc="left")
    return _save_figure(figure, output_dir, "paired_effects", summary_sha256)


def create_figures(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    _configure_style()
    summary = _read_json(summary_path)
    summary_sha256 = _sha256(summary_path)
    figures = {
        "training_curves": {
            "files": plot_training_curves(summary, output_dir, summary_sha256),
            "caption": (
                "Mean per-interaction training reward in "
                f"{summary.get('training_curve_bin_size', 2000)}-interaction bins. "
                "Bands are nonparametric 95% confidence intervals across ten "
                "independent training seeds."
            ),
        },
        "heldout_performance": {
            "files": plot_heldout_performance(summary, output_dir, summary_sha256),
            "caption": (
                "Held-out performance. Hollow marks show the ten independent "
                "training-seed means for learned policies or ten paired environment "
                "seeds for baselines; solid marks and bars show means and bootstrap "
                "95% confidence intervals."
            ),
        },
        "paired_effects": {
            "files": plot_paired_effects(summary, output_dir, summary_sha256),
            "caption": (
                "Held-out paired effects relative to the deterministic minimum-delay "
                "baseline. Intervals are bootstrapped across ten independent "
                "training-seed mean differences."
            ),
        },
    }
    manifest = {
        "format_version": 2,
        "protocol_version": "1.2",
        "source_summary": str(summary_path),
        "source_summary_sha256": summary_sha256,
        "figures": figures,
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else summary_path.parent / "figures"
    )
    create_figures(summary_path, output_dir)


if __name__ == "__main__":
    main()
