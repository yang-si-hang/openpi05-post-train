from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as _model
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
