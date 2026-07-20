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
DEFAULT_SUMMARY = (
    REPOSITORY / "results/formal/protocol_1_1_10seed_v2/summary/formal_summary.json"
)
ALGORITHM_LABELS = {
    "mixed_action_ppo": "Mixed-action PPO",
    "mixed_action_td3": "Mixed-action TD3",
    "minimum_estimated_delay": "Min. estimated delay",
    "random_masked": "Masked random",
}
COLORS = {
    "mixed_action_ppo": "#0072B2",
    "mixed_action_td3": "#D55E00",
    "minimum_estimated_delay": "#222222",
    "random_masked": "#777777",
}
MARKERS = {
    "mixed_action_ppo": "o",
    "mixed_action_td3": "s",
    "minimum_estimated_delay": "D",
    "random_masked": "^",
}


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
            "svg.hashsalt": "uav-mec-formal-protocol-1-1",
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
    for algorithm in ("mixed_action_ppo", "mixed_action_td3"):
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
    if policy in ("mixed_action_ppo", "mixed_action_td3"):
        values = summary["replicate_means"][policy]["heldout"][metric]
        stats = summary["summaries"]["candidates"][policy]["heldout"][metric]
    else:
        stats = summary["summaries"]["baselines"][policy]["heldout"][metric]
        values = stats["values"]
    return np.asarray(values, dtype=float), stats


def plot_heldout_performance(
    summary: Mapping[str, Any], output_dir: Path, summary_sha256: str
) -> list[str]:
    metrics = (
        ("total_reward", "Episode reward", 1.0),
        ("success_rate", "Task success (%)", 100.0),
        ("latency_per_settled_task_seconds", "Latency per settled task (s)", 1.0),
        ("energy_per_settled_task_joules", "Energy per settled task (J)", 1.0),
    )
    policies = (
        "minimum_estimated_delay",
        "random_masked",
        "mixed_action_ppo",
        "mixed_action_td3",
    )
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 5.15), constrained_layout=True)
    rng = np.random.default_rng(20260718)
    for axis, (metric, ylabel, scale) in zip(axes.flat, metrics, strict=True):
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
    axes[0, 0].set_title("Held-out performance: points and bootstrap 95% CI", loc="left")
    return _save_figure(
        figure, output_dir, "heldout_performance", summary_sha256
    )


def plot_paired_effects(
    summary: Mapping[str, Any], output_dir: Path, summary_sha256: str
) -> list[str]:
    metrics = (
        ("total_reward", "Reward difference"),
        ("success_rate", "Success-rate difference (percentage points)"),
        ("latency_per_settled_task_seconds", "Latency/task difference (s)"),
        ("energy_per_settled_task_joules", "Energy/task difference (J)"),
    )
    algorithms = ("mixed_action_ppo", "mixed_action_td3")
    figure, axes = plt.subplots(2, 2, figsize=(7.15, 4.65), constrained_layout=True)
    for axis, (metric, xlabel) in zip(axes.flat, metrics, strict=True):
        scale = 100.0 if metric == "success_rate" else 1.0
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
                "Mean per-interaction training reward in 256-interaction bins. "
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
        "format_version": 1,
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
