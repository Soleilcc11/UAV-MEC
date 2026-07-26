from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from uav_mec_gym.ppo import InvalidActionMaskError
from uav_mec_gym.td3 import MixedActionTD3, TD3Config


def _config(**overrides):
    values = {
        "hidden_sizes": (8,),
        "batch_size": 2,
        "replay_capacity": 16,
        "learning_starts": 2,
        "policy_delay": 2,
        "normalize_observations": False,
        "policy_noise": 0.1,
        "noise_clip": 0.2,
        "exploration_noise": 0.05,
        "discrete_exploration": 0.2,
    }
    values.update(overrides)
    return TD3Config(**values)


def _observation(*, terminal=False, offset=0.0, mask=None):
    if mask is None:
        mask = [0, 1, 1, 1]
    return {
        "time": np.array([offset], dtype=np.float32),
        "delta_time": np.array([offset], dtype=np.float32),
        "task": np.full(7, 0.1 + offset, dtype=np.float32),
        "resources": np.full((2, 3), 0.2 + offset, dtype=np.float32),
        "uavs": np.full((2, 8), 0.3 + offset, dtype=np.float32),
        "action_mask": np.zeros(4, dtype=np.int8)
        if terminal
        else np.asarray(mask, dtype=np.int8),
    }


def _transition(agent, index, *, terminal=False, truncated=False):
    observation = _observation(offset=0.01 * index)
    sample = agent.sample_action(
        observation, deterministic=False, update_normalizer=True
    )
    next_observation = _observation(
        terminal=terminal or truncated, offset=0.01 * (index + 1)
    )
    agent.store_transition(
        sample,
        reward=1.0 + index,
        next_observation=next_observation,
        terminated=terminal,
        truncated=truncated,
    )
    return sample, agent.update()


def test_config_requires_replay_and_warmup_to_cover_a_batch():
    with pytest.raises(ValueError, match="replay_capacity"):
        TD3Config(batch_size=8, replay_capacity=4)
    with pytest.raises(ValueError, match="learning_starts"):
        TD3Config(batch_size=8, replay_capacity=8, learning_starts=4)


def test_deterministic_action_obeys_mask_and_movement_bounds():
    agent = MixedActionTD3(2, _config(), seed=7)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.target_head.bias.copy_(
            torch.tensor([100.0, 3.0, 2.0, 1.0])
        )

    action = agent.act(_observation(mask=[0, 1, 1, 0]))

    assert action["target"] == 1
    assert np.asarray(action["movement"]).shape == (2, 3)
    assert np.max(np.abs(action["movement"])) <= 1.0


def test_all_masked_observation_is_rejected():
    agent = MixedActionTD3(2, _config(), seed=7)
    with pytest.raises(InvalidActionMaskError):
        agent.act(_observation(mask=[0, 0, 0, 0]))


def test_straight_through_target_has_hard_forward_and_zero_masked_gradient():
    agent = MixedActionTD3(2, _config(), seed=11)
    logits = torch.tensor(
        [[9.0, 2.0, 1.0, -1.0, -2.0]], requires_grad=True
    )
    mask = torch.tensor([[False, True, True, False, False]])

    target = agent._straight_through_target_vector(logits, mask)
    loss = (target * torch.arange(5, dtype=torch.float32)).sum()
    loss.backward()

    torch.testing.assert_close(
        target.detach(), torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    )
    assert logits.grad is not None
    assert logits.grad[0, 0].item() == 0.0
    assert logits.grad[0, 3].item() == 0.0
    assert logits.grad[0, 4].item() == 0.0
    assert logits.grad[0, 1].item() != 0.0


def test_twin_critic_updates_before_delayed_actor_and_handles_terminal_mask():
    agent = MixedActionTD3(2, _config(), seed=13)
    target_actor_before = {
        key: value.detach().clone()
        for key, value in agent.target_actor.state_dict().items()
    }

    first_sample, first_update = _transition(agent, 0)
    _, second_update = _transition(agent, 1, terminal=True)

    assert first_sample.warmup_random is True
    assert first_update is None
    assert second_update is not None
    assert second_update["actor_updated"] == 0.0
    assert second_update["actor_loss"] is None
    assert second_update["actor_grad_norm"] is None
    assert agent.update_count == 1
    assert agent.actor_update_count == 0
    for key, before in target_actor_before.items():
        torch.testing.assert_close(agent.target_actor.state_dict()[key], before)

    _, third_update = _transition(agent, 2)

    assert third_update is not None
    assert third_update["actor_updated"] == 1.0
    assert np.isfinite(third_update["actor_loss"])
    assert np.isfinite(third_update["actor_grad_norm"])
    assert agent.update_count == 2
    assert agent.actor_update_count == 1
    assert any(
        not torch.equal(target_actor_before[key], value)
        for key, value in agent.target_actor.state_dict().items()
    )


def test_time_limit_transition_without_next_action_does_not_bootstrap():
    agent = MixedActionTD3(2, _config(bootstrap_truncated=True), seed=15)

    _transition(agent, 0)
    _, update = _transition(agent, 1, truncated=True)

    assert update is not None
    assert np.isfinite(update["critic_loss"])


def test_nonterminal_transition_without_next_action_is_rejected():
    agent = MixedActionTD3(2, _config(), seed=16)
    _transition(agent, 0)
    observation = _observation(offset=0.1)
    sample = agent.sample_action(
        observation, deterministic=False, update_normalizer=True
    )
    agent.store_transition(
        sample,
        reward=1.0,
        next_observation=_observation(terminal=True, offset=0.2),
        terminated=False,
        truncated=False,
    )

    with pytest.raises(InvalidActionMaskError, match="non-terminal"):
        agent.update()


def test_checkpoint_resume_preserves_replay_rng_and_next_update(tmp_path: Path):
    agent = MixedActionTD3(2, _config(), seed=17)
    for index in range(4):
        _transition(agent, index, terminal=index % 2 == 1)
    checkpoint = tmp_path / "td3.pt"
    agent.save_checkpoint(checkpoint)
    resumed = MixedActionTD3.load_checkpoint(checkpoint)

    expected_sample = agent.sample_action(
        _observation(offset=0.5), deterministic=False, update_normalizer=True
    )
    actual_sample = resumed.sample_action(
        _observation(offset=0.5), deterministic=False, update_normalizer=True
    )
    assert actual_sample.target == expected_sample.target
    assert actual_sample.warmup_random == expected_sample.warmup_random
    np.testing.assert_array_equal(actual_sample.movement, expected_sample.movement)

    for candidate, sample in (
        (agent, expected_sample),
        (resumed, actual_sample),
    ):
        candidate.store_transition(
            sample,
            reward=2.5,
            next_observation=_observation(offset=0.6),
            terminated=False,
            truncated=False,
        )
    expected_update = agent.update()
    actual_update = resumed.update()

    assert actual_update == expected_update
    assert resumed.training_steps == agent.training_steps
    assert resumed.update_count == agent.update_count
    assert resumed.actor_update_count == agent.actor_update_count
    assert len(resumed.replay_buffer) == len(agent.replay_buffer)
    for expected_state, actual_state in (
        (agent.actor.state_dict(), resumed.actor.state_dict()),
        (agent.critic.state_dict(), resumed.critic.state_dict()),
        (agent.target_actor.state_dict(), resumed.target_actor.state_dict()),
        (agent.target_critic.state_dict(), resumed.target_critic.state_dict()),
    ):
        for key in expected_state:
            torch.testing.assert_close(
                actual_state[key], expected_state[key], rtol=0, atol=0
            )


def test_protocol_1_1_td3_checkpoint_is_rejected(tmp_path: Path):
    agent = MixedActionTD3(2, _config(), seed=18)
    checkpoint = tmp_path / "td3-1.2.pt"
    legacy = tmp_path / "td3-1.1.pt"
    agent.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format_version"] = 1
    payload["protocol_version"] = "1.1"
    torch.save(payload, legacy)

    with pytest.raises(ValueError, match="1.1"):
        MixedActionTD3.load_checkpoint(legacy)


def test_training_seed_cursor_is_checkpointed_and_rejects_a_new_schedule(
    tmp_path: Path,
):
    agent = MixedActionTD3(2, _config(), seed=19)
    assert agent.reserve_training_episode_seed(1001) == 1001
    checkpoint = tmp_path / "seed.pt"
    agent.save_checkpoint(checkpoint)
    resumed = MixedActionTD3.load_checkpoint(checkpoint)

    assert resumed.next_training_seed == 1002
    assert resumed.reserve_training_episode_seed(1001) == 1002
    with pytest.raises(ValueError, match="checkpointed training schedule"):
        resumed.reserve_training_episode_seed(2001)
