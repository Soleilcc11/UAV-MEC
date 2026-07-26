from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from uav_mec_gym import BackendStep, RewardComponents, UAVMECGymEnv


class DeterministicContractBackend:
    """Protocol fixture; never use this backend for training or evaluation."""

    def __init__(self, number_of_uavs: int):
        self.number_of_uavs = number_of_uavs
        self.step_count = 0
        self.seed = None
        self.reset_seeds = []

    def reset(self, seed):
        self.seed = seed
        self.reset_seeds.append(seed)
        self.step_count = 0
        return self._observation(), {"seed": seed}

    def step(self, target, movement):
        self.step_count += 1
        components = RewardComponents(
            success=1.0,
            latency_ratio=0.5,
            ue_energy_ratio=0.1,
            uav_energy_ratio=float(np.linalg.norm(movement)) / 10.0,
            constraint_violations=0.0 if target < 2 + self.number_of_uavs else 1.0,
        )
        return BackendStep(
            observation=self._observation(),
            reward_components=components,
            reward=UAVMECGymEnv.calculate_reward(components),
            terminated=self.step_count >= 2,
            truncated=False,
            metrics={"target": target},
        )

    def close(self):
        pass

    def _observation(self):
        target_count = 2 + self.number_of_uavs
        return {
            "time": np.array([min(self.step_count / 2.0, 1.0)], dtype=np.float32),
            "delta_time": np.array([0.5], dtype=np.float32),
            "task": np.full(7, 0.25, dtype=np.float32),
            "resources": np.full((2, 3), 0.5, dtype=np.float32),
            "uavs": np.full((self.number_of_uavs, 8), 0.5, dtype=np.float32),
            "action_mask": np.ones(target_count, dtype=np.int8),
        }


def test_contract_passes_gymnasium_checker():
    env = UAVMECGymEnv(2, DeterministicContractBackend(2))
    check_env(env)
    observation, _ = env.reset(seed=42)
    continuous_count = sum(
        int(np.asarray(observation[key]).size)
        for key in ("time", "delta_time", "task", "resources", "uavs")
    )
    assert continuous_count == 31
    assert observation["action_mask"].shape == (4,)


def test_reward_is_bounded_and_penalizes_costs():
    good = RewardComponents(1.0, 0.1, 0.1, 0.1, 0.0)
    bad = RewardComponents(0.0, 2.0, 2.0, 2.0, 1.0)

    assert -1.0 <= UAVMECGymEnv.calculate_reward(bad) <= 1.0
    assert UAVMECGymEnv.calculate_reward(good) > UAVMECGymEnv.calculate_reward(bad)


def test_environment_preserves_aggregate_backend_reward():
    backend = DeterministicContractBackend(1)
    env = UAVMECGymEnv(1, backend)
    env.reset(seed=1)
    backend.step = lambda target, movement: BackendStep(
        observation=backend._observation(),
        reward_components=RewardComponents(2.0, 0.0, 0.0, 0.0, 0.0),
        reward=2.0,
        terminated=True,
        truncated=False,
        metrics={"settled_in_transition": 2},
    )
    _, reward, terminated, _, info = env.step(
        {"target": 0, "movement": np.zeros((1, 3), dtype=np.float32)}
    )
    assert reward == 2.0
    assert terminated is True
    assert info["settled_in_transition"] == 2


def test_unseeded_resets_derive_a_reproducible_episode_seed_sequence():
    def collect_sequence():
        backend = DeterministicContractBackend(1)
        env = UAVMECGymEnv(1, backend)
        infos = [env.reset(seed=2026)[1], env.reset()[1], env.reset()[1]]
        seeds = [int(info["seed"]) for info in infos]
        assert backend.reset_seeds == seeds
        return seeds

    first = collect_sequence()
    second = collect_sequence()

    assert first == second
    assert first[0] == 2026
    assert len(set(first)) == len(first)


def test_explicit_seed_restarts_the_unseeded_episode_seed_stream():
    backend = DeterministicContractBackend(1)
    env = UAVMECGymEnv(1, backend)

    env.reset(seed=77)
    first_derived_seed = env.reset()[1]["seed"]
    env.reset(seed=77)
    repeated_derived_seed = env.reset()[1]["seed"]

    assert first_derived_seed == repeated_derived_seed


def test_reset_rejects_a_backend_that_omits_the_actual_seed():
    class MissingSeedBackend(DeterministicContractBackend):
        def reset(self, seed):
            observation, _ = super().reset(seed)
            return observation, {}

    env = UAVMECGymEnv(1, MissingSeedBackend(1))

    with pytest.raises(ValueError, match="omitted the actual episode seed"):
        env.reset(seed=42)
