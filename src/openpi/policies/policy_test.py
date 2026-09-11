from types import SimpleNamespace

import numpy as np
from openpi_client import action_chunk_broker
import pytest

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize
from openpi.training import config as _config


def test_prepare_rtc_inputs_reuses_relative_action_pipeline():
    action_horizon = 6
    model_action_dim = 5
    delta_mask = (True, False, True)
    state = np.array([1.0, 10.0, 3.0], dtype=np.float32)
    absolute_actions = np.array([[2.0, 20.0, 6.0], [4.0, 30.0, 8.0]], dtype=np.float32)
    action_stats = normalize.NormStats(
        mean=np.array([0.5, 2.0, 1.0], dtype=np.float32),
        std=np.array([2.0, 4.0, 5.0], dtype=np.float32),
    )
    transform = transforms.compose(
        [
            transforms.DeltaActions(delta_mask),
            transforms.Normalize({"actions": action_stats}),
            transforms.PadStatesAndActions(model_action_dim),
        ]
    )

    transformed = transform({"state": state.copy(), "actions": absolute_actions.copy()})
    training_reference = transform({"state": state.copy(), "actions": absolute_actions.copy()})["actions"]
    np.testing.assert_allclose(transformed["actions"], training_reference)

    delta_reference = absolute_actions - np.where(delta_mask, state, 0.0)[None, :]
    normalized_reference = (delta_reference - action_stats.mean) / (action_stats.std + 1e-6)
    np.testing.assert_allclose(transformed["actions"][:, :3], normalized_reference)
    # The non-delta dimension remains an absolute target before normalization.
    np.testing.assert_allclose(transformed["actions"][:, 1], (absolute_actions[:, 1] - 2.0) / (4.0 + 1e-6))

    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = SimpleNamespace(action_horizon=action_horizon, action_dim=model_action_dim)  # noqa: SLF001
    policy._supports_train_time_rtc = False  # noqa: SLF001
    fixed_actions, weights, max_weight, use_vjp, info, prefix_len = policy._prepare_rtc_inputs(  # noqa: SLF001
        transformed,
        {
            "mode": "vjp",
            "prev_actions_abs": absolute_actions,
            "prefix_len": 1,
            "decay_end": 4,
            "schedule": "linear",
            "max_guidance_weight": 5.0,
        },
        "vjp",
    )

    np.testing.assert_allclose(fixed_actions[:2], training_reference)
    np.testing.assert_array_equal(fixed_actions[2:], np.zeros((4, model_action_dim), dtype=np.float32))
    np.testing.assert_array_equal(weights[2:], np.zeros(4, dtype=np.float32))
    assert max_weight == np.float32(5.0)
    assert use_vjp is True
    assert prefix_len == 1
    assert info == {
        "mode": "vjp",
        "enabled": True,
        "prefix_len": 1,
        "prev_action_steps": 2,
    }


def test_prepare_rtc_inputs_rejects_transform_that_drops_actions():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = SimpleNamespace(action_horizon=6, action_dim=5)  # noqa: SLF001
    policy._supports_train_time_rtc = False  # noqa: SLF001

    with pytest.raises(ValueError, match="input transforms dropped"):
        policy._prepare_rtc_inputs(  # noqa: SLF001
            {"state": np.zeros(5, dtype=np.float32)},
            {"mode": "non_vjp", "prev_actions_abs": np.zeros((2, 3), dtype=np.float32), "prefix_len": 1},
            "non_vjp",
        )


def test_explicit_mode_rejects_legacy_use_vjp():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._supports_train_time_rtc = False  # noqa: SLF001

    with pytest.raises(ValueError, match=r"rtc\.use_vjp is not allowed"):
        policy._resolve_rtc_mode(  # noqa: SLF001
            {"mode": "vjp", "prev_actions_abs": np.zeros((2, 3)), "prefix_len": 1, "use_vjp": True}
        )


def test_missing_mode_defaults_to_off_and_rejects_rtc_fields():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._supports_train_time_rtc = False  # noqa: SLF001

    assert policy._resolve_rtc_mode(None) == "off"  # noqa: SLF001
    assert policy._resolve_rtc_mode({}) == "off"  # noqa: SLF001
    with pytest.raises(ValueError, match=r"not allowed when rtc\.mode is off"):
        policy._resolve_rtc_mode({"prefix_len": 1})  # noqa: SLF001


def test_checkpoint_mode_conflicts_are_rejected():
    policy = _policy.Policy.__new__(_policy.Policy)

    policy._supports_train_time_rtc = False  # noqa: SLF001
    with pytest.raises(ValueError, match="requires a checkpoint configured"):
        policy._resolve_rtc_mode({"mode": "train_rtc"})  # noqa: SLF001

    policy._supports_train_time_rtc = True  # noqa: SLF001
    with pytest.raises(ValueError, match="cannot be used with a training-time RTC checkpoint"):
        policy._resolve_rtc_mode({"mode": "non_vjp"})  # noqa: SLF001
    assert policy._resolve_rtc_mode({"mode": "train_rtc"}) == "train_rtc"  # noqa: SLF001


def test_prepare_train_rtc_inputs_uses_hard_prefix_without_weights():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = SimpleNamespace(action_horizon=6, action_dim=5, train_time_rtc_max_delay=3)  # noqa: SLF001
    policy._supports_train_time_rtc = True  # noqa: SLF001
    actions = np.ones((3, 5), dtype=np.float32)

    fixed, weights, max_weight, use_vjp, info, prefix_len = policy._prepare_rtc_inputs(  # noqa: SLF001
        {"state": np.zeros(5, dtype=np.float32), "actions": actions},
        {"mode": "train_rtc", "prev_actions_abs": actions, "prefix_len": 2},
        "train_rtc",
    )

    np.testing.assert_array_equal(fixed[:3], actions)
    np.testing.assert_array_equal(weights, np.zeros(6, dtype=np.float32))
    assert max_weight == 0
    assert use_vjp is False
    assert prefix_len == 2
    assert info == {
        "mode": "train_rtc",
        "enabled": True,
        "prefix_len": 2,
        "prev_action_steps": 3,
    }


def test_train_rtc_rejects_guidance_fields_and_invalid_prefixes():
    policy = _policy.Policy.__new__(_policy.Policy)
    policy._model = SimpleNamespace(action_horizon=6, action_dim=5, train_time_rtc_max_delay=3)  # noqa: SLF001
    policy._supports_train_time_rtc = True  # noqa: SLF001

    with pytest.raises(ValueError, match="does not support"):
        policy._resolve_rtc_mode({"mode": "train_rtc", "schedule": "exp"})  # noqa: SLF001

    for prefix_len, error_type, match in [
        (True, TypeError, "must be an integer"),
        (4, ValueError, "exceeds the checkpoint max delay"),
        (3, ValueError, "exceeds the available previous actions"),
    ]:
        actions = np.ones((2, 5), dtype=np.float32)
        with pytest.raises(error_type, match=match):
            policy._prepare_rtc_inputs(  # noqa: SLF001
                {"state": np.zeros(5, dtype=np.float32), "actions": actions},
                {"mode": "train_rtc", "prev_actions_abs": actions, "prefix_len": prefix_len},
                "train_rtc",
            )


def test_ur_train_time_rtc_config_is_explicit_and_metadata_has_no_checkpoint_capability():
    standard = _config.get_config("pi05_ur10e_lora_finetune")
    train_rtc = _config.get_config("pi05_ur10e_lora_train_time_rtc")

    assert standard.model.train_time_rtc_max_delay == 0
    assert train_rtc.model.train_time_rtc_max_delay == 10
    assert train_rtc.model.action_horizon == 20
    assert train_rtc.batch_size == 1
    assert train_rtc.num_train_steps == 30_000
    assert train_rtc.policy_metadata == {
        "prediction_horizon": 20,
        "execution_horizon": 10,
        "action_dim": 10,
        "control_frequency_hz": 20,
    }


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "gs://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)
