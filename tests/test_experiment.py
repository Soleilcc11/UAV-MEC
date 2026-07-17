from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from uav_mec_gym.evaluation import MinimumEstimatedDelayPolicy, RandomMaskedPolicy
from uav_mec_gym.experiment import (
    build_fair_evaluation_report,
    environment_manifest,
    train_ppo,
)
from uav_mec_gym.ppo import MixedActionPPO, PPOConfig


class _ShortEpisodeEnv:
    def __init__(self) -> None:
        self.steps = 0
        self.seed = 0

    def reset(self, *, seed=None, options=None):
        del options
        self.steps = 0
        self.seed = int(seed)
        return self._observation(), {"seed": self.seed}

    def step(self, action):
        assert int(action["target"]) in (1, 2, 3, 4)
        assert np.asarray(action["movement"]).shape == (2, 3)
        self.steps += 1
        terminated = self.steps >= 2
        success = float(int(action["target"]) == 2)
        return (
            self._observation(terminal=terminated),
            0.5 + success,
            terminated,
            False,
            {
                "settled_tasks": self.steps,
                "total_tasks": 2,
                "simulation_time": float(self.steps),
                "reward_components": {
                    "success": success,
                    "latency_ratio": 0.25,
                    "ue_energy_ratio": 0.1,
                    "uav_energy_ratio": 0.05,
                    "constraint_violations": 0.0,
                },
                "latency_seconds": 1.0,
                "deadline_seconds": 4.0,
                "ue_energy_joules": 0.2,
                "uav_energy_joules": 0.3,
                "constraint_violations": 0.0,
            },
        )

    def _observation(self, terminal=False):
        return {
            "time": np.array([1.0 if terminal else self.steps / 2], dtype=np.float32),
            "task": np.zeros(7, dtype=np.float32)
            if terminal
            else np.full(7, 0.25, dtype=np.float32),
            "resources": np.full((3, 3), 0.4, dtype=np.float32),
            "uavs": np.full((2, 8), 0.3, dtype=np.float32),
            "action_mask": np.zeros(5, dtype=np.int8)
            if terminal
            else np.array([0, 1, 1, 1, 1], dtype=np.int8),
        }


def _agent(seed=7):
    return MixedActionPPO(
        2,
        PPOConfig(
            hidden_sizes=(8,),
            rollout_steps=2,
            minibatch_size=2,
            update_epochs=1,
            normalize_observations=False,
        ),
        seed=seed,
    )


def test_training_uses_exact_interaction_budget_and_flushes_tail():
    agent = _agent()

    episodes, updates = train_ppo(
        _ShortEpisodeEnv(),
        agent,
        interaction_budget=3,
        training_seed_start=101,
    )

    assert agent.training_steps == 3
    assert sum(episode["steps"] for episode in episodes) == 3
    assert episodes[0]["seed"] == 101
    assert episodes[0]["terminated"] is True
    assert episodes[1]["seed"] == 102
    assert episodes[1]["truncated"] is True
    assert episodes[1]["transitions"][-1]["interaction_budget_truncated"] is True
    assert len(agent.buffer) == 0
    assert sum(int(update["rollout_steps"]) for update in updates) == 3


def test_environment_manifest_is_content_addressed(tmp_path: Path):
    settings = tmp_path / "simulation_settings.xml"
    applications = tmp_path / "applications.xml"
    settings.write_text("<settings/>", encoding="utf-8")
    applications.write_text("<applications/>", encoding="utf-8")

    first = environment_manifest([settings, applications])
    second = environment_manifest([applications, settings])

    assert first == second
    assert len(first["sha256"]) == 64
    assert {entry["name"] for entry in first["files"]} == {
        "simulation_settings.xml",
        "applications.xml",
    }


def test_fair_report_records_paired_seeds_hashes_and_raw_metrics(tmp_path: Path):
    checkpoint = tmp_path / "ppo.pt"
    agent = _agent(seed=11)
    agent.save_checkpoint(checkpoint)
    policies = [RandomMaskedPolicy(2), MinimumEstimatedDelayPolicy(2), agent]

    report = build_fair_evaluation_report(
        _ShortEpisodeEnv(),
        policies,
        seeds=[301, 302, 303, 304, 305],
        split="heldout",
        environment={"files": [], "sha256": "environment-hash"},
        commit_sha="abc123",
        checkpoint_path=checkpoint,
    )

    assert report["paired_seeds"] == [301, 302, 303, 304, 305]
    assert len(report["episodes"]) == 15
    assert report["audit"]["status"] == "passed"
    assert report["training_interactions"]["mixed_action_ppo"] == 0
    assert "minimum_estimated_delay" in report["paired_candidate_minus_reference"]
    for row in report["episodes"]:
        assert row["git_commit_sha"] == "abc123"
        assert row["environment_config_hash"] == "environment-hash"
        assert row["physical_metric_sums"]["latency_seconds"] == 2.0
        if row["policy"] == "mixed_action_ppo":
            assert row["checkpoint_sha256"] == report["checkpoint"]["sha256"]
            assert row["training_interactions"] == 0
        else:
            assert row["checkpoint"] is None


def test_fair_report_rejects_fewer_than_five_paired_seeds(tmp_path: Path):
    checkpoint = tmp_path / "ppo.pt"
    agent = _agent()
    agent.save_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="five unique paired seeds"):
        build_fair_evaluation_report(
            _ShortEpisodeEnv(),
            [RandomMaskedPolicy(2), agent],
            seeds=[1, 2, 3, 4],
            split="heldout",
            environment={"files": [], "sha256": "hash"},
            commit_sha="abc123",
            checkpoint_path=checkpoint,
        )
