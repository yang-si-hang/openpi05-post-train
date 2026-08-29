from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_rtc_disabled_matches_plain_sampling():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=5,
    )
    model = config.create(key)
    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(jax.random.key(1), (1, config.action_horizon, config.action_dim))
    sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_use_vjp",))

    plain = sample_actions(key, obs, num_steps=2, noise=noise)
    rtc_disabled = sample_actions(
        key,
        obs,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=jnp.zeros_like(noise),
        rtc_prefix_weights=jnp.zeros((config.action_horizon,), dtype=noise.dtype),
        rtc_max_guidance_weight=jnp.asarray(5.0),
        rtc_use_vjp=False,
    )

    np.testing.assert_allclose(rtc_disabled, plain, rtol=1e-7, atol=1e-7)

    rtc_enabled = sample_actions(
        key,
        obs,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=jnp.zeros_like(noise),
        rtc_prefix_weights=jnp.array([1.0, 0.5, 0.0, 0.0, 0.0], dtype=noise.dtype),
        rtc_max_guidance_weight=jnp.asarray(5.0),
        rtc_use_vjp=False,
    )
    assert rtc_enabled.shape == plain.shape
    assert np.all(np.isfinite(rtc_enabled))
    assert not np.allclose(rtc_enabled, plain)

    rtc_vjp = sample_actions(
        key,
        obs,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=jnp.zeros_like(noise),
        rtc_prefix_weights=jnp.array([1.0, 0.5, 0.0, 0.0, 0.0], dtype=noise.dtype),
        rtc_max_guidance_weight=jnp.asarray(5.0),
        rtc_use_vjp=True,
    )
    assert rtc_vjp.shape == plain.shape
    assert np.all(np.isfinite(rtc_vjp))


def test_train_time_rtc_corruption_and_loss_normalization():
    actions = jnp.arange(2 * 5 * 3, dtype=jnp.float32).reshape(2, 5, 3)
    noise = -actions
    time = jnp.array([0.25, 0.75], dtype=jnp.float32)
    delay = jnp.array([0, 3], dtype=jnp.int32)

    x_t, token_time, prefix_mask = pi0.apply_train_time_rtc_corruption(actions, noise, time, delay)

    np.testing.assert_array_equal(prefix_mask[0], np.zeros(5, dtype=bool))
    np.testing.assert_array_equal(prefix_mask[1], np.array([True, True, True, False, False]))
    np.testing.assert_array_equal(token_time[1, :3], np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(x_t[1, :3], actions[1, :3])

    per_step_loss = jnp.ones((2, 5), dtype=jnp.float32)
    normalized = pi0.normalize_postfix_loss(per_step_loss, prefix_mask)
    np.testing.assert_allclose(jnp.mean(normalized), 1.0, rtol=1e-6)


def test_train_time_rtc_delay_sampling_includes_both_endpoints():
    delay = pi0.sample_train_time_rtc_delay(jax.random.key(123), (4096,), max_delay=4)
    assert int(jnp.min(delay)) == 0
    assert int(jnp.max(delay)) == 4


def test_pi05_train_time_rtc_hard_prefix_sampling():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=5,
        train_time_rtc_max_delay=3,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    prev_actions = jax.random.normal(jax.random.key(2), noise.shape)
    prefix_len = jnp.array([0, 2], dtype=jnp.int32)

    loss = nnx_utils.module_jit(model.compute_loss)(jax.random.key(3), obs, prev_actions)
    assert loss.shape == (2, config.action_horizon)
    assert np.all(np.isfinite(loss))

    sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_use_vjp",))
    result = sample_actions(
        jax.random.key(4),
        obs,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=prev_actions,
        rtc_prefix_len=prefix_len,
        rtc_use_vjp=False,
    )

    np.testing.assert_array_equal(result[1, :2], prev_actions[1, :2])
    assert np.all(np.isfinite(result))

    plain = sample_actions(jax.random.key(4), obs, num_steps=2, noise=noise)
    zero_prefix = sample_actions(
        jax.random.key(4),
        obs,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=prev_actions,
        rtc_prefix_len=jnp.zeros((2,), dtype=jnp.int32),
        rtc_use_vjp=False,
    )
    np.testing.assert_array_equal(zero_prefix, plain)


def test_train_time_rtc_preserves_parameter_tree_and_adarms_supports_token_conditioning():
    common = {
        "pi05": True,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "action_dim": 4,
        "action_horizon": 5,
    }
    plain_model = pi0_config.Pi0Config(**common).create(jax.random.key(0))
    rtc_model = pi0_config.Pi0Config(**common, train_time_rtc_max_delay=3).create(jax.random.key(0))
    assert jax.tree.structure(nnx.state(plain_model)) == jax.tree.structure(nnx.state(rtc_model))
    plain_shapes = [leaf.shape for leaf in jax.tree.leaves(nnx.state(plain_model))]
    rtc_shapes = [leaf.shape for leaf in jax.tree.leaves(nnx.state(rtc_model))]
    assert plain_shapes == rtc_shapes

    norm = gemma.RMSNorm()
    tokens = jnp.ones((2, 5, 8), dtype=jnp.float32)
    global_cond = jnp.ones((2, 8), dtype=jnp.float32)
    variables = norm.init(jax.random.key(1), tokens, global_cond)
    global_output, global_gate = norm.apply(variables, tokens, global_cond)
    token_output, token_gate = norm.apply(variables, tokens, jnp.ones((2, 5, 8), dtype=jnp.float32))
    assert global_output.shape == token_output.shape == tokens.shape
    assert global_gate.shape == (2, 1, 8)
    assert token_gate.shape == tokens.shape


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
