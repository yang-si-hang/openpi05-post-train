import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _make_ki_model_and_batch():
    config = _pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=2,
        max_token_len=8,
        knowledge_insulation=True,
        ki_max_token_len=8,
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    # A single SigLIP patch per camera keeps this source-graph test lightweight.
    observation = observation.replace(images={name: image[:, :14, :14] for name, image in observation.images.items()})
    return model, observation, config.fake_act(batch_size=2)


def _reference_encode_images(model: _pi0.Pi0, observation: _model.Observation) -> _pi0.EncodedImages:
    """Preserve the pre-refactor per-camera image concatenation as an independent reference."""
    tokens = []
    token_masks = []
    ar_masks = []
    for name in observation.images:
        camera_tokens, _ = model.PaliGemma.img(observation.images[name], train=False)
        tokens.append(camera_tokens)
        token_masks.append(jnp.repeat(observation.image_masks[name][:, None], camera_tokens.shape[1], axis=1))
        ar_masks.append(jnp.zeros((camera_tokens.shape[1],), dtype=jnp.bool_))
    return _pi0.EncodedImages(
        tokens=jnp.concatenate(tokens, axis=1),
        token_mask=jnp.concatenate(token_masks, axis=1),
        ar_mask=jnp.concatenate(ar_masks, axis=0),
    )


def test_ki_encoded_images_and_losses_match_independent_reference(monkeypatch):
    model, observation, actions = _make_ki_model_and_batch()

    image_module = model.PaliGemma.img
    wrapper_type = type(image_module)
    original_call = wrapper_type.__call__

    def lightweight_image_call(self, image, *args, **kwargs):
        if self is not image_module:
            return original_call(self, image, *args, **kwargs)
        pooled = jnp.mean(image, axis=(1, 2, 3), keepdims=False)
        tokens = jnp.broadcast_to(pooled[:, None, None], (image.shape[0], 2, 64))
        return tokens, {}

    monkeypatch.setattr(wrapper_type, "__call__", lightweight_image_call)

    reference_images = _reference_encode_images(model, observation)
    encoded_images = model._encode_images(observation)  # noqa: SLF001
    np.testing.assert_allclose(encoded_images.tokens, reference_images.tokens, rtol=0, atol=0)
    np.testing.assert_array_equal(encoded_images.token_mask, reference_images.token_mask)
    np.testing.assert_array_equal(encoded_images.ar_mask, reference_images.ar_mask)
    assert encoded_images.tokens.shape[:2] == encoded_images.token_mask.shape
    assert encoded_images.ar_mask.shape == (encoded_images.tokens.shape[1],)

    noise_rng, time_rng = jax.random.split(jax.random.key(1))
    reference_fast, reference_correct, reference_count = model._compute_fast_loss(observation)  # noqa: SLF001
    reference_flow = model._compute_flow_loss_preprocessed(  # noqa: SLF001
        noise_rng, time_rng, None, observation, actions
    )
    shared_fast, shared_correct, shared_count = model._compute_fast_loss(  # noqa: SLF001
        observation, encoded_images=encoded_images
    )
    shared_flow = model._compute_flow_loss_preprocessed(  # noqa: SLF001
        noise_rng, time_rng, None, observation, actions, encoded_images=encoded_images
    )

    np.testing.assert_allclose(shared_fast, reference_fast, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(shared_flow, reference_flow, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(shared_correct, reference_correct)
    np.testing.assert_array_equal(shared_count, reference_count)

    reference_total = model.ki_fast_loss_weight * jnp.mean(reference_fast) + model.ki_flow_loss_weight * jnp.mean(
        reference_flow
    )
    shared_total = model.ki_fast_loss_weight * jnp.mean(shared_fast) + model.ki_flow_loss_weight * jnp.mean(shared_flow)
    np.testing.assert_allclose(shared_total, reference_total, rtol=1e-5, atol=1e-5)


def test_ki_compute_loss_encodes_images_once(monkeypatch):
    model, observation, actions = _make_ki_model_and_batch()
    call_count = 0

    def fake_encode_images(_self, obs):
        nonlocal call_count
        call_count += 1
        batch_size = obs.state.shape[0]
        width = 64
        num_image_tokens = len(obs.images)
        return _pi0.EncodedImages(
            tokens=jnp.zeros((batch_size, num_image_tokens, width), dtype=jnp.float32),
            token_mask=jnp.ones((batch_size, num_image_tokens), dtype=jnp.bool_),
            ar_mask=jnp.zeros((num_image_tokens,), dtype=jnp.bool_),
        )

    monkeypatch.setattr(_pi0.Pi0, "_encode_images", fake_encode_images)
    total, _ = model.compute_loss_with_aux(jax.random.key(2), observation, actions, train=False)

    assert np.isfinite(total)
    assert call_count == 1
