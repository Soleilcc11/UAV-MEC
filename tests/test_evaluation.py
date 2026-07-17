import numpy as np

from uav_mec_gym.evaluation import MinimumEstimatedDelayPolicy, RandomMaskedPolicy


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
