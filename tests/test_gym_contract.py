from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from uav_mec_gym import BackendStep, RewardComponents, UAVMECGymEnv


class DeterministicContractBackend:
    """Protocol fixture; never use this backend for training or evaluation."""

    def __init__(self, number_of_uavs: int):
        self.number_of_uavs = number_of_uavs
        self.step_count = 0
        self.seed = None

    def reset(self, seed):
        self.seed = seed
        self.step_count = 0
        return self._observation(), {"seed": seed}

    def step(self, target, movement):
        self.step_count += 1
        components = RewardComponents(
            success=1.0,
            latency_ratio=0.5,
            ue_energy_ratio=0.1,
            uav_energy_ratio=float(np.linalg.norm(movement)) / 10.0,
            constraint_violations=0.0 if target < 3 + self.number_of_uavs else 1.0,
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
        target_count = 3 + self.number_of_uavs
        return {
            "time": np.array([min(self.step_count / 2.0, 1.0)], dtype=np.float32),
            "task": np.full(7, 0.25, dtype=np.float32),
            "resources": np.full((3, 3), 0.5, dtype=np.float32),
            "uavs": np.full((self.number_of_uavs, 8), 0.5, dtype=np.float32),
            "action_mask": np.ones(target_count, dtype=np.int8),
        }


def test_contract_passes_gymnasium_checker():
    env = UAVMECGymEnv(2, DeterministicContractBackend(2))
    check_env(env)


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
