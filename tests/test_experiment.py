from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from uav_mec_gym.evaluation import (
    MinimumEstimatedDelayPolicy,
    RandomMaskedPolicy,
    evaluate_policy,
)
from uav_mec_gym.experiment import (
    audit_episode_results,
    build_fair_evaluation_report,
    environment_manifest,
    repository_source_sha256,
    train_td3,
    train_ppo,
)
from uav_mec_gym.ppo import MixedActionPPO, PPOConfig
from uav_mec_gym.td3 import MixedActionTD3, TD3Config


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
        assert int(action["target"]) in (1, 2, 3)
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
                "settled_in_transition": 1,
                "total_tasks": 2,
                "simulation_time": float(self.steps),
                "elapsed_simulation_time": 1.0,
                "effective_discount": 0.99,
                "selected_cloud_relay_uav": -1,
                "throughput_tasks_per_second": success / float(self.steps),
                "uav_queue_length_total": 0,
                "uav_queue_length_max": 0,
                "active_access_uploads": 0,
                "active_access_downloads": 0,
                "active_backhaul_uploads": 0,
                "active_backhaul_downloads": 0,
                "local_resource_utilization": 0.1,
                "cloud_resource_utilization": 0.2,
                "uav_resource_utilization": 0.3,
                "settled_task_latencies_seconds": [1.0],
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
            "delta_time": np.array([0.5], dtype=np.float32),
            "task": np.zeros(7, dtype=np.float32)
            if terminal
            else np.full(7, 0.25, dtype=np.float32),
            "resources": np.full((2, 3), 0.4, dtype=np.float32),
            "uavs": np.full((2, 8), 0.3, dtype=np.float32),
            "action_mask": np.zeros(4, dtype=np.int8)
            if terminal
            else np.array([0, 1, 1, 1], dtype=np.int8),
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


def _td3_agent(seed=7):
    return MixedActionTD3(
        2,
        TD3Config(
            hidden_sizes=(8,),
            batch_size=2,
            replay_capacity=16,
            learning_starts=2,
            policy_delay=2,
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
    assert agent.training_seed_start == 101
    assert agent.training_episode_count == 2
    assert agent.next_training_seed == 103


def test_td3_training_uses_exact_budget_and_records_online_updates():
    agent = _td3_agent()

    episodes, updates = train_td3(
        _ShortEpisodeEnv(),
        agent,
        interaction_budget=3,
        training_seed_start=1001,
    )

    assert agent.training_steps == 3
    assert sum(episode["steps"] for episode in episodes) == 3
    assert [episode["seed"] for episode in episodes] == [1001, 1002]
    assert episodes[-1]["truncated"] is True
    assert episodes[-1]["transitions"][-1]["interaction_budget_truncated"] is True
    assert episodes[0]["transitions"][0]["warmup_random"] is True
    assert len(updates) == 2
    assert updates[0]["actor_loss"] is None
    assert updates[0]["actor_grad_norm"] is None
    json.dumps(updates, allow_nan=False)
    assert agent.update_count == 2
    assert agent.actor_update_count == 1
    assert agent.next_training_seed == 1003


def test_checkpoint_resume_matches_uninterrupted_training_and_advances_seeds(
    tmp_path: Path,
):
    uninterrupted = _agent(seed=101)
    uninterrupted_episodes, _ = train_ppo(
        _ShortEpisodeEnv(),
        uninterrupted,
        interaction_budget=4,
        training_seed_start=101,
    )

    split = _agent(seed=101)
    first_episodes, _ = train_ppo(
        _ShortEpisodeEnv(),
        split,
        interaction_budget=2,
        training_seed_start=101,
    )
    checkpoint = tmp_path / "resume.pt"
    split.save_checkpoint(checkpoint)
    resumed = MixedActionPPO.load_checkpoint(checkpoint)
    resumed_episodes, _ = train_ppo(
        _ShortEpisodeEnv(),
        resumed,
        interaction_budget=4,
        training_seed_start=101,
    )

    assert [episode["seed"] for episode in uninterrupted_episodes] == [101, 102]
    assert [episode["seed"] for episode in first_episodes] == [101]
    assert [episode["seed"] for episode in resumed_episodes] == [102]
    assert [
        transition["global_step"]
        for episode in resumed_episodes
        for transition in episode["transitions"]
    ] == [3, 4]
    assert resumed.training_steps == uninterrupted.training_steps == 4
    assert resumed.update_count == uninterrupted.update_count == 2
    assert resumed.training_seed_start == uninterrupted.training_seed_start == 101
    assert resumed.training_episode_count == uninterrupted.training_episode_count == 2
    assert resumed.next_training_seed == uninterrupted.next_training_seed == 103

    expected_transitions = [
        transition
        for episode in uninterrupted_episodes
        for transition in episode["transitions"]
    ]
    resumed_transitions = [
        transition
        for episode in first_episodes + resumed_episodes
        for transition in episode["transitions"]
    ]
    for expected, actual in zip(expected_transitions, resumed_transitions, strict=True):
        assert actual["target"] == expected["target"]
        assert actual["old_joint_log_prob"] == expected["old_joint_log_prob"]
        np.testing.assert_array_equal(actual["movement"], expected["movement"])

    for expected_state, actual_state in (
        (uninterrupted.actor.state_dict(), resumed.actor.state_dict()),
        (uninterrupted.critic.state_dict(), resumed.critic.state_dict()),
    ):
        assert expected_state.keys() == actual_state.keys()
        for key in expected_state:
            torch.testing.assert_close(
                expected_state[key], actual_state[key], rtol=0, atol=0
            )


def test_resume_requires_a_larger_cumulative_target_and_same_seed_schedule():
    agent = _agent(seed=101)
    train_ppo(
        _ShortEpisodeEnv(),
        agent,
        interaction_budget=2,
        training_seed_start=101,
    )

    with pytest.raises(ValueError, match="cumulative target"):
        train_ppo(
            _ShortEpisodeEnv(),
            agent,
            interaction_budget=2,
            training_seed_start=101,
        )
    with pytest.raises(ValueError, match="checkpointed training schedule"):
        train_ppo(
            _ShortEpisodeEnv(),
            agent,
            interaction_budget=4,
            training_seed_start=999,
        )


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


def test_repository_source_hash_tracks_relative_paths_and_contents(tmp_path: Path):
    (tmp_path / "src/main/java/example").mkdir(parents=True)
    (tmp_path / "src/main/resources").mkdir(parents=True)
    source = tmp_path / "src/main/java/example/Main.java"
    source.write_text("class Main {}", encoding="utf-8")
    first = repository_source_sha256(tmp_path)
    source.write_text("class Main { int value; }", encoding="utf-8")
    second = repository_source_sha256(tmp_path)
    assert first != second


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


def test_episode_audit_tolerates_one_ulp_mean_roundoff():
    agent = _agent(seed=19)
    episode_results = evaluate_policy(_ShortEpisodeEnv(), agent, [401])
    result = episode_results[0]
    maximum = result.audit_metric_maxima["cloud_resource_utilization"]
    result.audit_metric_means["cloud_resource_utilization"] = float(
        np.nextafter(maximum, np.inf)
    )

    audit_episode_results(episode_results)


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


def test_fair_report_accepts_td3_as_the_single_learned_candidate(tmp_path: Path):
    checkpoint = tmp_path / "td3.pt"
    agent = _td3_agent(seed=31)
    agent.save_checkpoint(checkpoint)

    report = build_fair_evaluation_report(
        _ShortEpisodeEnv(),
        [RandomMaskedPolicy(2), MinimumEstimatedDelayPolicy(2), agent],
        seeds=list(range(301, 311)),
        split="heldout",
        environment={"files": [], "sha256": "environment-hash"},
        commit_sha="abc123",
        checkpoint_path=checkpoint,
    )

    assert report["checkpoint"]["sha256"]
    assert report["training_interactions"]["mixed_action_td3"] == 0
    assert "minimum_estimated_delay" in report["paired_candidate_minus_reference"]
    candidate_rows = [
        row for row in report["episodes"] if row["policy"] == "mixed_action_td3"
    ]
    assert len(candidate_rows) == 10
    assert all(row["checkpoint_sha256"] for row in candidate_rows)
    assert report["audit"]["interpretation"].startswith("descriptive smoke evidence")
