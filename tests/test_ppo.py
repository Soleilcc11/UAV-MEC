from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from uav_mec_gym.ppo import (
    MixedActionPPO,
    PPOConfig,
    compute_gae,
    masked_categorical,
)


def _observation(
    *,
    action_mask: Sequence[int] = (1, 1, 1, 1),
    offset: float = 0.0,
) -> dict[str, np.ndarray]:
    """Small observation satisfying the frozen two-UAV Gymnasium contract."""
    return {
        "time": np.array([0.2 + offset], dtype=np.float32),
        "delta_time": np.array([0.1 + offset], dtype=np.float32),
        "task": np.linspace(0.1, 0.7, 7, dtype=np.float32) + offset,
        "resources": np.full((2, 3), 0.4 + offset, dtype=np.float32),
        "uavs": np.full((2, 8), 0.3 + offset, dtype=np.float32),
        "action_mask": np.asarray(action_mask, dtype=np.int8),
    }


def _config(**overrides: Any) -> PPOConfig:
    settings: dict[str, Any] = {
        "hidden_sizes": (16,),
        "rollout_steps": 32,
        "minibatch_size": 8,
        "update_epochs": 1,
        "learning_rate": 1e-3,
        "normalize_observations": True,
        "normalize_advantages": False,
        "device": "cpu",
    }
    settings.update(overrides)
    return PPOConfig(**settings)


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(np.asarray(value).item())


def _assert_nested_equal(left: Any, right: Any) -> None:
    """Compare optimizer/normalizer state without depending on their internals."""
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def _add_rollout(agent: MixedActionPPO, length: int = 3) -> None:
    for index in range(length):
        observation = _observation(offset=0.01 * index)
        next_observation = _observation(offset=0.01 * (index + 1))
        sample = agent.sample_action(observation, update_normalizer=True)
        agent.store_transition(
            sample,
            reward=(1.0, 0.25, -0.5)[index % 3],
            next_observation=next_observation,
            terminated=index == length - 1,
            truncated=False,
        )


def test_masked_target_distribution_has_exact_zero_probability_for_illegal_actions():
    logits = torch.tensor([[2.0, 100.0, -1.0, 50.0, 0.5]])
    mask = torch.tensor([[1, 0, 1, 0, 1]], dtype=torch.bool)

    distribution = masked_categorical(logits, mask)

    torch.testing.assert_close(
        distribution.probs[0, [1, 3]],
        torch.zeros(2),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(distribution.probs.sum(dim=-1), torch.ones(1))


def test_sampling_never_selects_a_masked_target_and_all_masked_fails_explicitly():
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=7)
    observation = _observation(action_mask=(0, 1, 0, 1))

    sampled_targets = {
        int(agent.sample_action(observation).action["target"]) for _ in range(128)
    }

    assert sampled_targets <= {1, 3}
    with pytest.raises(ValueError, match=r"(?i)(mask|valid|legal|enabled)"):
        agent.sample_action(_observation(action_mask=(0, 0, 0, 0)))


def test_observation_contract_rejects_missing_keys_and_wrong_shapes():
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=8)
    missing = _observation()
    del missing["task"]
    wrong_shape = _observation()
    wrong_shape["uavs"] = np.zeros((1, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="frozen Dict contract"):
        agent.sample_action(missing)
    with pytest.raises(ValueError, match="expected"):
        agent.sample_action(wrong_shape)


@pytest.mark.parametrize("deterministic", [False, True])
def test_movement_has_frozen_shape_and_bounds(deterministic: bool):
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=11)

    for _ in range(16):
        action = agent.act(_observation(), deterministic=deterministic)
        movement = np.asarray(action["movement"])
        assert movement.shape == (2, 3)
        assert np.all(movement >= -1.0)
        assert np.all(movement <= 1.0)


def test_stored_old_log_probability_is_joint_probability_of_realized_action():
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=19)
    observation = _observation(action_mask=(1, 0, 1, 1))
    sample = agent.sample_action(observation)

    expected_joint = _scalar(sample.target_log_prob) + _scalar(
        sample.movement_log_prob
    )
    assert _scalar(sample.log_prob) == pytest.approx(expected_joint, abs=1e-6)

    agent.store_transition(
        sample,
        reward=0.5,
        next_observation=_observation(action_mask=(1, 1, 0, 1), offset=0.01),
        terminated=False,
        truncated=False,
    )

    # This is the PPO ratio's old log-prob. In particular it must not be the
    # distribution entropy (the legacy implementation accidentally stored it).
    assert _scalar(agent.buffer.log_probs[-1]) == pytest.approx(
        expected_joint, abs=1e-6
    )
    assert int(agent.buffer.targets[-1]) == int(sample.action["target"])
    np.testing.assert_allclose(
        np.asarray(agent.buffer.movements[-1]),
        np.asarray(sample.action["movement"]),
        rtol=0,
        atol=0,
    )


def test_saturated_policy_sends_interior_float32_action_used_by_log_probability():
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=20)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        agent.actor.movement_mean_head.bias.fill_(20.0)

    sample = agent.sample_action(_observation(), deterministic=True)
    movement = np.asarray(sample.action["movement"])

    assert np.all(movement < 1.0)
    assert np.all(movement > -1.0)
    assert np.isfinite(sample.movement_log_prob)


def test_gae_never_bootstraps_a_terminated_transition():
    advantages, returns = compute_gae(
        rewards=np.array([1.0]),
        values=np.array([0.5]),
        next_values=np.array([10.0]),
        terminated=np.array([True]),
        truncated=np.array([False]),
        gamma=0.9,
        gae_lambda=0.95,
        bootstrap_truncated=True,
    )

    np.testing.assert_allclose(advantages, [0.5])
    np.testing.assert_allclose(returns, [1.0])


@pytest.mark.parametrize(
    ("bootstrap_truncated", "expected_advantage", "expected_return"),
    [(True, 9.5, 10.0), (False, 0.5, 1.0)],
)
def test_gae_truncation_bootstrap_is_configurable(
    bootstrap_truncated: bool,
    expected_advantage: float,
    expected_return: float,
):
    advantages, returns = compute_gae(
        rewards=np.array([1.0]),
        values=np.array([0.5]),
        next_values=np.array([10.0]),
        terminated=np.array([False]),
        truncated=np.array([True]),
        gamma=0.9,
        gae_lambda=0.95,
        bootstrap_truncated=bootstrap_truncated,
    )

    np.testing.assert_allclose(advantages, [expected_advantage])
    np.testing.assert_allclose(returns, [expected_return])


def test_gae_uses_per_transition_smdp_discounts():
    advantages, returns = compute_gae(
        rewards=np.array([0.0, 1.0]),
        values=np.array([0.0, 0.0]),
        next_values=np.array([1.0, 0.0]),
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
        gamma=0.99,
        discounts=np.array([0.5, 0.25]),
        gae_lambda=1.0,
    )

    np.testing.assert_allclose(advantages, [1.0, 1.0])
    np.testing.assert_allclose(returns, [1.0, 1.0])


def test_checkpoint_round_trip_preserves_training_state_normalizer_and_rng(
    tmp_path: Path,
):
    agent = MixedActionPPO(2, _config(), seed=23)
    _add_rollout(agent)
    assert agent.update() is not None  # populate Adam moments and counters

    checkpoint = tmp_path / "ppo.pt"
    agent.save_checkpoint(checkpoint)

    expected_optimizer = agent.optimizer.state_dict()
    expected_normalizer = agent.observation_normalizer.state_dict()
    expected_training_steps = agent.training_steps
    expected_update_count = agent.update_count
    expected_python = agent.python_rng.random()
    expected_numpy = agent.numpy_rng.random()
    expected_torch = torch.rand(4, generator=agent.torch_rng)
    expected_stochastic = agent.sample_action(_observation())
    expected_deterministic = agent.act(_observation(), deterministic=True)

    restored = MixedActionPPO.load_checkpoint(checkpoint)

    _assert_nested_equal(expected_optimizer, restored.optimizer.state_dict())
    _assert_nested_equal(
        expected_normalizer, restored.observation_normalizer.state_dict()
    )
    assert restored.training_steps == expected_training_steps
    assert restored.update_count == expected_update_count
    assert restored.python_rng.random() == expected_python
    assert restored.numpy_rng.random() == expected_numpy
    torch.testing.assert_close(
        torch.rand(4, generator=restored.torch_rng), expected_torch, rtol=0, atol=0
    )

    restored_stochastic = restored.sample_action(_observation())
    assert restored_stochastic.action["target"] == expected_stochastic.action["target"]
    np.testing.assert_array_equal(
        restored_stochastic.action["movement"],
        expected_stochastic.action["movement"],
    )
    assert _scalar(restored_stochastic.log_prob) == pytest.approx(
        _scalar(expected_stochastic.log_prob), abs=1e-7
    )

    restored_deterministic = restored.act(_observation(), deterministic=True)
    assert restored_deterministic["target"] == expected_deterministic["target"]
    np.testing.assert_array_equal(
        restored_deterministic["movement"], expected_deterministic["movement"]
    )


def test_protocol_1_1_ppo_checkpoint_is_rejected(tmp_path: Path):
    agent = MixedActionPPO(2, _config(), seed=91)
    checkpoint = tmp_path / "ppo-1.2.pt"
    legacy = tmp_path / "ppo-1.1.pt"
    agent.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["format_version"] = 2
    payload["protocol_version"] = "1.1"
    torch.save(payload, legacy)

    with pytest.raises(ValueError, match="1.1"):
        MixedActionPPO.load_checkpoint(legacy)


def test_checkpoint_restores_partial_rollout(tmp_path: Path):
    agent = MixedActionPPO(2, _config(), seed=24)
    _add_rollout(agent, length=2)
    checkpoint = tmp_path / "partial.pt"
    agent.save_checkpoint(checkpoint)

    restored = MixedActionPPO.load_checkpoint(checkpoint)

    assert len(restored.buffer) == 2
    assert restored.training_steps == 2
    np.testing.assert_array_equal(
        restored.buffer.movements[0], agent.buffer.movements[0]
    )
    assert restored.buffer.log_probs == agent.buffer.log_probs


def test_policy_reset_replays_the_same_seeded_sampling_stream():
    agent = MixedActionPPO(2, _config(normalize_observations=False), seed=25)
    agent.reset(777)
    first = agent.sample_action(_observation())
    agent.reset(777)
    second = agent.sample_action(_observation())

    assert first.action["target"] == second.action["target"]
    np.testing.assert_array_equal(first.action["movement"], second.action["movement"])
    assert first.log_prob == pytest.approx(second.log_prob, abs=0.0)


def test_update_consumes_a_rollout_smaller_than_the_minibatch():
    agent = MixedActionPPO(
        2,
        _config(minibatch_size=8, update_epochs=2, normalize_observations=False),
        seed=29,
    )
    _add_rollout(agent, length=3)
    parameters_before = [
        parameter.detach().clone()
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]

    assert len(agent.buffer) == 3
    metrics = agent.update()
    parameters_after = [
        parameter.detach()
        for module in (agent.actor, agent.critic)
        for parameter in module.parameters()
    ]

    assert metrics is not None
    assert len(agent.buffer) == 0
    assert agent.update_count == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(parameters_before, parameters_after)
    )
