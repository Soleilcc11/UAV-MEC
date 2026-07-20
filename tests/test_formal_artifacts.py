from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.plot_formal_results import create_figures
from scripts.run_formal_experiments import _validate_config
from scripts.summarize_formal_experiments import bootstrap_summary, metric_value


def test_bootstrap_summary_is_seeded_and_uses_sample_standard_deviation():
    first = bootstrap_summary([1.0, 2.0, 3.0, 4.0], seed=7, resamples=1000)
    second = bootstrap_summary([1.0, 2.0, 3.0, 4.0], seed=7, resamples=1000)

    assert first == second
    assert first["mean"] == 2.5
    assert first["sample_std"] == pytest.approx(1.2909944487)
    assert first["bootstrap_95_ci_low"] <= first["mean"]
    assert first["bootstrap_95_ci_high"] >= first["mean"]


def test_metric_value_preserves_physical_units_and_normalizes_per_settled_task():
    row = {
        "total_reward": 12.5,
        "successful_tasks": 3,
        "settled_tasks": 4,
        "physical_metric_sums": {
            "latency_seconds": 8.0,
            "ue_energy_joules": 4.0,
            "uav_energy_joules": 12.0,
            "constraint_violations": 2.0,
        },
    }

    assert metric_value(row, "total_reward") == 12.5
    assert metric_value(row, "success_rate") == 0.75
    assert metric_value(row, "latency_per_settled_task_seconds") == 2.0
    assert metric_value(row, "energy_per_settled_task_joules") == 4.0
    assert metric_value(row, "constraint_violations") == 2.0


def _stats(values):
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "sample_std": 0.1,
        "median": mean,
        "bootstrap_95_ci_low": mean - 0.1,
        "bootstrap_95_ci_high": mean + 0.1,
        "bootstrap_resamples": 100,
        "values": values,
    }


def _synthetic_summary():
    metrics = (
        "total_reward",
        "success_rate",
        "latency_per_settled_task_seconds",
        "energy_per_settled_task_joules",
        "constraint_violations",
    )
    algorithms = ("mixed_action_ppo", "mixed_action_td3")
    baselines = ("minimum_estimated_delay", "random_masked")
    payload = {
        "audit": {"status": "passed"},
        "training_curves": {},
        "replicate_means": {},
        "summaries": {"candidates": {}, "baselines": {}, "paired": {}},
    }
    for algorithm_index, algorithm in enumerate(algorithms):
        payload["training_curves"][algorithm] = {
            "summary": [
                {"interaction_end": 256, **_stats([1.0 + algorithm_index] * 10)},
                {"interaction_end": 512, **_stats([2.0 + algorithm_index] * 10)},
            ]
        }
        payload["replicate_means"][algorithm] = {"heldout": {}}
        payload["summaries"]["candidates"][algorithm] = {"heldout": {}}
        payload["summaries"]["paired"][algorithm] = {"heldout": {}}
        for metric_index, metric in enumerate(metrics):
            values = [float(metric_index + algorithm_index + index / 10) for index in range(10)]
            payload["replicate_means"][algorithm]["heldout"][metric] = values
            payload["summaries"]["candidates"][algorithm]["heldout"][metric] = _stats(values)
            payload["summaries"]["paired"][algorithm]["heldout"][metric] = {
                baseline: _stats([value - 0.5 for value in values])
                for baseline in baselines
            }
    for baseline_index, baseline in enumerate(baselines):
        payload["summaries"]["baselines"][baseline] = {"heldout": {}}
        for metric_index, metric in enumerate(metrics):
            payload["summaries"]["baselines"][baseline]["heldout"][metric] = _stats(
                [float(metric_index + baseline_index + index / 20) for index in range(10)]
            )
    return payload


def test_plotter_emits_vector_and_300_dpi_raster_figures(tmp_path: Path):
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_synthetic_summary()), encoding="utf-8")
    output = tmp_path / "figures"

    manifest = create_figures(summary_path, output)

    assert set(manifest["figures"]) == {
        "training_curves",
        "heldout_performance",
        "paired_effects",
    }
    for figure in manifest["figures"].values():
        assert len(figure["files"]) == 3
        for raw_path in figure["files"]:
            path = Path(raw_path)
            assert path.stat().st_size > 1000
            if path.suffix == ".png":
                with Image.open(path) as image:
                    assert image.info["dpi"][0] == pytest.approx(300, rel=0.01)
                    assert image.width >= 1500


def test_formal_config_rejects_fewer_than_ten_training_seeds():
    config = {
        "format_version": 1,
        "protocol_version": "1.1",
        "interaction_budget": 4096,
        "algorithms": {
            "mixed_action_td3": {
                "command": "train-td3",
                "training_seed_starts": list(range(9)),
            }
        },
        "evaluation_seeds": {
            "validation": list(range(10)),
            "heldout": list(range(10, 20)),
        },
        "environments": {
            "training_validation": ["missing-a", "missing-b", "missing-c"],
            "heldout": ["missing-d", "missing-e", "missing-f"],
        },
    }
    with pytest.raises(ValueError, match="ten unique training"):
        _validate_config(config)
