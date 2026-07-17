import numpy as np
import pytest

from uav_mec_gym.evaluation import (
    EpisodeResult,
    MinimumEstimatedDelayPolicy,
    RandomMaskedPolicy,
    evaluate_policy,
    summarize_episode_metric,
    summarize_paired_seed_differences,
    summarize_values,
)


def _observation():
    return {
        "action_mask": np.array([0, 1, 1, 1, 0], dtype=np.int8),
        "resources": np.array([[0, 0, 0], [1, 0, 0.4], [1, 0, 0.2]], dtype=np.float32),
        "uavs": np.array([
            [0, 0, 0, 1, 0, 1, 0.05, 0.05],
            [0, 0, 0, 1, 0, 1, 0.01, 0.01],
        ], dtype=np.float32),
    }


def test_random_policy_is_seeded_and_respects_mask():
    policy = RandomMaskedPolicy(2)
    policy.reset(42)
    first = policy.act(_observation())
    policy.reset(42)
    second = policy.act(_observation())
    assert first["target"] in (1, 2, 3)
    assert first["target"] == second["target"]
    np.testing.assert_array_equal(first["movement"], second["movement"])


def test_delay_policy_selects_best_enabled_observable_target():
    action = MinimumEstimatedDelayPolicy(2).act(_observation())
    assert action["target"] == 3
    np.testing.assert_array_equal(action["movement"], np.zeros((2, 3)))


class _TwoStepEnvironment:
    def reset(self, *, seed):
        self.seed = seed
        self.step_number = 0
        return _observation(), {"seed": seed}

    def step(self, action):
        del action
        frames = [
            {
                "reward": 0.4,
                "latency_seconds": 1.0,
                "deadline_seconds": 4.0,
                "ue_energy_joules": 0.25,
                "uav_energy_joules": 2.0,
                "constraint_violations": 0,
                "success": 1.0,
                "settled_tasks": 1,
                "simulation_time": 2.0,
            },
            {
                "reward": -0.2,
                "latency_seconds": 3.0,
                "deadline_seconds": 4.0,
                "ue_energy_joules": 0.75,
                "uav_energy_joules": 4.0,
                "constraint_violations": 2,
                "success": 0.0,
                "settled_tasks": 2,
                "simulation_time": 5.0,
            },
        ]
        frame = frames[self.step_number]
        self.step_number += 1
        done = self.step_number == len(frames)
        info = {
            key: frame[key]
            for key in (
                "latency_seconds",
                "deadline_seconds",
                "ue_energy_joules",
                "uav_energy_joules",
                "constraint_violations",
                "settled_tasks",
                "simulation_time",
            )
        }
        info["total_tasks"] = 2
        info["reward_components"] = {
            "success": frame["success"],
            "latency_ratio": frame["latency_seconds"] / frame["deadline_seconds"],
            "ue_energy_ratio": frame["ue_energy_joules"] / 10.0,
            "uav_energy_ratio": frame["uav_energy_joules"] / 100.0,
            "constraint_violations": float(frame["constraint_violations"] > 0),
        }
        terminated = done and self.seed % 2 == 1
        truncated = done and self.seed % 2 == 0
        return _observation(), frame["reward"], terminated, truncated, info


def test_evaluate_policy_aggregates_raw_metrics_and_preserves_episode_status():
    results = evaluate_policy(
        _TwoStepEnvironment(), MinimumEstimatedDelayPolicy(2), [7, 8]
    )

    assert [result.seed for result in results] == [7, 8]
    assert (results[0].terminated, results[0].truncated) == (True, False)
    assert (results[1].terminated, results[1].truncated) == (False, True)
    assert results[0].steps == 2
    assert results[0].total_reward == pytest.approx(0.2)
    assert results[0].successful_tasks == 1
    assert results[0].physical_metric_sums == {
        "latency_seconds": 4.0,
        "deadline_seconds": 8.0,
        "ue_energy_joules": 1.0,
        "uav_energy_joules": 6.0,
        "constraint_violations": 2.0,
    }
    assert results[0].physical_metric_means == {
        "latency_seconds": 2.0,
        "deadline_seconds": 4.0,
        "ue_energy_joules": 0.5,
        "uav_energy_joules": 3.0,
        "constraint_violations": 1.0,
    }


def _episode(policy, seed, reward, latency):
    return EpisodeResult(
        policy=policy,
        seed=seed,
        steps=2,
        total_reward=reward,
        terminated=True,
        truncated=False,
        settled_tasks=2,
        total_tasks=2,
        simulation_time=5.0,
        successful_tasks=1,
        reward_component_sums={},
        physical_metric_sums={"latency_seconds": latency},
        physical_metric_means={"latency_seconds": latency / 2.0},
    )


def test_summary_reports_sample_std_and_normal_95_percent_ci():
    summary = summarize_values([1.0, 2.0, 3.0, 4.0])

    expected_std = np.std([1.0, 2.0, 3.0, 4.0], ddof=1)
    expected_margin = 1.96 * expected_std / np.sqrt(4)
    assert summary.count == 4
    assert summary.mean == pytest.approx(2.5)
    assert summary.std == pytest.approx(expected_std)
    assert summary.ci95_lower == pytest.approx(2.5 - expected_margin)
    assert summary.ci95_upper == pytest.approx(2.5 + expected_margin)


def test_paired_seed_summary_matches_by_seed_and_supports_physical_metrics():
    references = [
        _episode("reference", 11, 1.0, 5.0),
        _episode("reference", 12, 3.0, 8.0),
    ]
    candidates = [
        _episode("candidate", 12, 5.0, 7.0),
        _episode("candidate", 11, 4.0, 9.0),
    ]

    paired = summarize_paired_seed_differences(references, candidates)
    latency = summarize_episode_metric(
        candidates, "physical_metric_means.latency_seconds"
    )

    assert paired.reference_policy == "reference"
    assert paired.candidate_policy == "candidate"
    assert paired.differences_by_seed == {11: 3.0, 12: 2.0}
    assert paired.statistics.mean == pytest.approx(2.5)
    assert paired.statistics.std == pytest.approx(np.sqrt(0.5))
    assert latency.mean == pytest.approx(4.0)


def test_paired_seed_summary_rejects_unfair_seed_sets():
    reference = [_episode("reference", 1, 1.0, 2.0)]
    candidate = [_episode("candidate", 2, 2.0, 2.0)]

    with pytest.raises(ValueError, match="identical seeds"):
        summarize_paired_seed_differences(reference, candidate)
