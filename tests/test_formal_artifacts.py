from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from scripts.package_experiment_artifacts import DEFAULT_ROOT
from scripts.plot_formal_results import DEFAULT_SUMMARY, create_figures
from scripts import run_formal_experiments
from scripts.run_formal_experiments import (
    _evaluation_is_complete,
    _sha256,
    _training_is_complete,
    _validate_config,
)
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


def test_publication_helpers_follow_the_configured_output_root():
    config = json.loads(
        run_formal_experiments.DEFAULT_CONFIG.read_text(encoding="utf-8")
    )
    expected = run_formal_experiments.REPOSITORY / config["output_root"]

    assert DEFAULT_ROOT == expected
    assert DEFAULT_SUMMARY == expected / "summary/formal_summary.json"


def test_dry_run_preserves_completed_orchestration_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repository = Path(run_formal_experiments.REPOSITORY)
    output_root = tmp_path / "formal"
    output_root.mkdir()
    manifest = output_root / "orchestration.json"
    original = b'{"status":"complete","sentinel":true}\n'
    manifest.write_bytes(original)
    config = {
        "format_version": 1,
        "experiment_id": "dry_run_regression",
        "protocol_version": "1.1",
        "number_of_uavs": 2,
        "interaction_budget": 1024,
        "output_root": str(output_root),
        "ports": {"training_validation": 12470, "heldout": 12471},
        "environments": {
            "training_validation": [
                str(repository / "src/test/resources/config/simulation_settings.xml"),
                str(repository / "src/test/resources/config/edge_devices.xml"),
                str(repository / "src/test/resources/config/applications.xml"),
            ],
            "heldout": [
                str(repository / "src/test/resources/config/heldout/simulation_settings.xml"),
                str(repository / "src/test/resources/config/heldout/edge_devices.xml"),
                str(repository / "src/test/resources/config/heldout/applications.xml"),
            ],
        },
        "evaluation_seeds": {
            "validation": list(range(201, 211)),
            "heldout": list(range(301, 311)),
        },
        "algorithms": {
            "mixed_action_td3": {
                "command": "train-td3",
                "evaluation_algorithm": "td3",
                "training_seed_starts": list(range(10001, 10011)),
            }
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(run_formal_experiments, "_require_clean_tracked_tree", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_formal_experiments.py",
            "--config",
            str(config_path),
            "--dry-run",
            "--skip-build",
            "--max-runs",
            "1",
        ],
    )

    run_formal_experiments.main()

    assert manifest.read_bytes() == original
    assert list(output_root.iterdir()) == [manifest]


def test_resume_requires_exact_training_and_evaluation_provenance(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    identity = {
        "experiment_id": "formal-v3",
        "config_sha256": "1" * 64,
        "git_commit_sha": "2" * 40,
        "environment_sha256": "3" * 64,
        "source_tree_sha256": "4" * 64,
    }
    shared = {
        "formal_experiment": {
            "experiment_id": identity["experiment_id"],
            "config_sha256": identity["config_sha256"],
        },
        "git_commit_sha": identity["git_commit_sha"],
        "environment": {"sha256": identity["environment_sha256"]},
        "service_provenance": {
            "runtime": {
                "git_commit_sha": identity["git_commit_sha"],
                "source_tree_sha256": identity["source_tree_sha256"],
                "classes_current": True,
            }
        },
        "checkpoint": {"sha256": _sha256(checkpoint)},
    }
    training_path = tmp_path / "training.json"
    training = {
        **shared,
        "algorithm": "mixed_action_td3",
        "interaction_budget": 4096,
        "starting_training_interactions": 0,
        "interactions_this_run": 4096,
        "seed_partitions": {"training_seeds_used": [110001]},
    }
    training_path.write_text(json.dumps(training), encoding="utf-8")
    training_arguments = {
        "algorithm": "mixed_action_td3",
        "interaction_budget": 4096,
        "training_seed_start": 110001,
        **identity,
    }
    assert _training_is_complete(training_path, checkpoint, **training_arguments)
    assert not _training_is_complete(
        training_path,
        checkpoint,
        **{**training_arguments, "training_seed_start": 999999},
    )
    assert not _training_is_complete(
        training_path,
        checkpoint,
        **{**training_arguments, "git_commit_sha": "5" * 40},
    )

    evaluation_path = tmp_path / "validation.json"
    evaluation = {
        **shared,
        "split": "validation",
        "paired_seeds": list(range(201, 211)),
        "audit": {"status": "passed"},
        "training_interactions": {"mixed_action_td3": 4096},
    }
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
    evaluation_arguments = {
        "algorithm": "mixed_action_td3",
        "split": "validation",
        "seeds": list(range(201, 211)),
        "interaction_budget": 4096,
        **identity,
    }
    assert _evaluation_is_complete(
        evaluation_path, checkpoint, **evaluation_arguments
    )
    assert not _evaluation_is_complete(
        evaluation_path,
        checkpoint,
        **{**evaluation_arguments, "environment_sha256": "6" * 64},
    )
