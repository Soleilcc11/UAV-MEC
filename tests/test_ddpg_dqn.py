from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from uav_mec_gym.ddpg import DDPGConfig, MixedActionDDPG
from uav_mec_gym.dqn import DQNConfig, MaskedDQN


def _observation(*, terminal: bool = False, offset: float = 0.0):
    return {
        "time": np.array([offset], dtype=np.float32),
        "delta_time": np.array([offset], dtype=np.float32),
        "task": np.full(7, 0.2 + offset, dtype=np.float32),
        "resources": np.full((2, 3), 0.3 + offset, dtype=np.float32),
        "uavs": np.full((2, 8), 0.4 + offset, dtype=np.float32),
        "action_mask": np.zeros(4, dtype=np.int8)
        if terminal
        else np.array([0, 1, 1, 1], dtype=np.int8),
    }


def test_optimized_ddpg_updates_single_critic_and_round_trips_checkpoint(
    tmp_path: Path,
):
    config = DDPGConfig(
        hidden_sizes=(8,), batch_size=2, replay_capacity=8,
        learning_starts=2, normalize_observations=False,
    )
    agent = MixedActionDDPG(2, config=config, seed=17)
    for index in range(2):
        sample = agent.sample_action(_observation(offset=0.01 * index))
        agent.store_transition(
            sample, 0.5, _observation(terminal=index == 1, offset=0.01 * (index + 1)),
            terminated=index == 1, truncated=False,
        )
    update = agent.update()

    assert update is not None
    assert update["actor_updated"] == 1.0
    assert "q2_mean" not in update
    checkpoint = tmp_path / "ddpg.pt"
    agent.save_checkpoint(checkpoint)
    restored = MixedActionDDPG.load_checkpoint(checkpoint)
    assert restored.training_steps == agent.training_steps
    assert restored.name == "masked_parameterized_action_ddpg"


def test_masked_dqn_uses_zero_movement_updates_and_round_trips_checkpoint(
    tmp_path: Path,
):
    config = DQNConfig(
        hidden_sizes=(8,), batch_size=2, replay_capacity=8,
        learning_starts=2, target_update_interval=1,
        normalize_observations=False,
    )
    agent = MaskedDQN(2, config=config, seed=23)
    for index in range(2):
        sample = agent.sample_action(_observation(offset=0.01 * index))
        np.testing.assert_array_equal(
            sample.action["movement"], np.zeros((2, 3), dtype=np.float32)
        )
        assert sample.action["target"] in (1, 2, 3)
        agent.store_transition(
            sample, 0.25, _observation(terminal=index == 1, offset=0.01 * (index + 1)),
            terminated=index == 1, truncated=False,
        )
    update = agent.update()

    assert update is not None
    checkpoint = tmp_path / "dqn.pt"
    agent.save_checkpoint(checkpoint)
    restored = MaskedDQN.load_checkpoint(checkpoint)
    assert restored.training_steps == agent.training_steps
    assert restored.name == "masked_dqn_zero_movement"


@pytest.mark.parametrize(
    ("agent_factory", "loader"),
    [
        (
            lambda: MixedActionDDPG(
                2,
                DDPGConfig(
                    hidden_sizes=(8,),
                    batch_size=2,
                    replay_capacity=8,
                    learning_starts=2,
                ),
                seed=31,
            ),
            MixedActionDDPG.load_checkpoint,
        ),
        (
            lambda: MaskedDQN(
                2,
                DQNConfig(
                    hidden_sizes=(8,),
                    batch_size=2,
                    replay_capacity=8,
                    learning_starts=2,
                ),
                seed=32,
            ),
            MaskedDQN.load_checkpoint,
        ),
    ],
)
def test_protocol_1_1_replay_checkpoint_is_rejected(
    tmp_path: Path, agent_factory, loader
):
    checkpoint = tmp_path / "current.pt"
    legacy = tmp_path / "legacy.pt"
    agent_factory().save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    payload["protocol_version"] = "1.1"
    torch.save(payload, legacy)

    with pytest.raises(ValueError, match="1.1"):
        loader(legacy)
