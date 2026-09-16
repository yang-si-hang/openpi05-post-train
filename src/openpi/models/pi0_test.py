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


def test_ki_shared_prefix_matches_independent_flow_reference(monkeypatch):
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
    _, reference_prefix_mask, reference_kv = model._forward_prefix(observation)  # noqa: SLF001
    reference_flow = model._compute_flow_loss_preprocessed(  # noqa: SLF001
        noise_rng, time_rng, None, observation, actions
    )
    prefix_out, prefix_mask, kv_common = model._forward_prefix(  # noqa: SLF001
        observation, encoded_images=encoded_images
    )
    shared_fast, shared_correct, shared_count = model._compute_fast_loss_from_shared_prefix(  # noqa: SLF001
        observation,
        prefix_out=prefix_out,
        prefix_mask=prefix_mask,
        kv_cache=kv_common,
    )
    shared_flow = model._compute_flow_loss_from_prefix_cache(  # noqa: SLF001
        noise_rng,
        time_rng,
        None,
        observation,
        actions,
        prefix_mask=prefix_mask,
        kv_cache=jax.tree.map(jax.lax.stop_gradient, kv_common),
    )

    np.testing.assert_array_equal(prefix_mask, reference_prefix_mask)
    for shared_leaf, reference_leaf in zip(jax.tree.leaves(kv_common), jax.tree.leaves(reference_kv), strict=True):
        np.testing.assert_allclose(
            np.asarray(shared_leaf, dtype=np.float32),
            np.asarray(reference_leaf, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
    np.testing.assert_allclose(shared_flow, reference_flow, rtol=1e-5, atol=1e-5)
    assert np.all(np.isfinite(shared_fast))
    assert np.isfinite(shared_correct)
    assert shared_count == observation.ki_action_token_mask.sum()


def test_ki_fast_incremental_matches_one_shot_with_mask_hole():
    model, observation, _ = _make_ki_model_and_batch()
    observation = observation.replace(
        ki_action_tokens=jnp.tile(jnp.arange(1, 9, dtype=jnp.int32), (2, 1)),
        ki_action_token_mask=jnp.asarray([[True, True, True, True, False, False, False, False]] * 2),
    )
    encoded_images = _pi0.EncodedImages(
        tokens=jnp.zeros((2, 3, 64), dtype=jnp.float32),
        token_mask=jnp.asarray([[True, False, True]] * 2),
        ar_mask=jnp.zeros((3,), dtype=jnp.bool_),
    )

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation, encoded_images=encoded_images)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
    )

    targets = observation.ki_action_tokens
    target_mask = observation.ki_action_token_mask
    suffix_input_ids = targets[:, :-1]
    suffix_input_mask = target_mask[:, :-1]
    suffix_embeddings = model.PaliGemma.llm(suffix_input_ids, method="embed")
    suffix_ar_mask = jnp.ones((suffix_input_ids.shape[1],), dtype=jnp.bool_)
    suffix_attn_mask = _pi0.make_attn_mask(suffix_input_mask, suffix_ar_mask)
    prefix_region = suffix_input_mask[:, :, None] & prefix_mask[:, None, :]
    incremental_mask = jnp.concatenate([prefix_region, suffix_attn_mask], axis=-1)
    suffix_positions = jnp.sum(prefix_mask, axis=1)[:, None] + jnp.cumsum(suffix_input_mask, axis=1) - 1
    (incremental_suffix_out, _), _ = model.PaliGemma.llm(
        [suffix_embeddings, None],
        kv_cache=kv_cache,
        mask=incremental_mask,
        positions=suffix_positions,
    )

    full_embeddings = jnp.concatenate([prefix_tokens, suffix_embeddings], axis=1)
    full_input_mask = jnp.concatenate([prefix_mask, suffix_input_mask], axis=1)
    full_ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    full_attn_mask = _pi0.make_attn_mask(full_input_mask, full_ar_mask)
    full_positions = jnp.cumsum(full_input_mask, axis=1) - 1
    (one_shot_out, _), _ = model.PaliGemma.llm([full_embeddings, None], mask=full_attn_mask, positions=full_positions)
    one_shot_prefix_out = one_shot_out[:, : prefix_tokens.shape[1]]
    one_shot_suffix_out = one_shot_out[:, prefix_tokens.shape[1] :]

    np.testing.assert_allclose(
        np.asarray(prefix_out, dtype=np.float32)[np.asarray(prefix_mask)],
        np.asarray(one_shot_prefix_out, dtype=np.float32)[np.asarray(prefix_mask)],
        rtol=2e-2,
        atol=2e-2,
    )
    valid_suffix = np.asarray(suffix_input_mask)
    np.testing.assert_allclose(
        np.asarray(incremental_suffix_out, dtype=np.float32)[valid_suffix],
        np.asarray(one_shot_suffix_out, dtype=np.float32)[valid_suffix],
        rtol=2e-2,
        atol=2e-2,
    )

    physical_indices = jnp.arange(prefix_mask.shape[1])[None, :]
    last_valid_index = jnp.max(jnp.where(prefix_mask, physical_indices, -1), axis=1)
    assert np.all(np.asarray(last_valid_index) > np.asarray(prefix_mask.sum(axis=1) - 1))
    incremental_last = jnp.take_along_axis(prefix_out, last_valid_index[:, None, None], axis=1)
    one_shot_last = jnp.take_along_axis(one_shot_prefix_out, last_valid_index[:, None, None], axis=1)
    incremental_prediction = jnp.concatenate([incremental_last, incremental_suffix_out], axis=1)
    one_shot_prediction = jnp.concatenate([one_shot_last, one_shot_suffix_out], axis=1)
    incremental_logits = model.PaliGemma.llm(incremental_prediction, method="decode")
    one_shot_logits = model.PaliGemma.llm(one_shot_prediction, method="decode")
    valid_targets = np.asarray(target_mask)
    np.testing.assert_allclose(
        np.asarray(incremental_logits, dtype=np.float32)[valid_targets],
        np.asarray(one_shot_logits, dtype=np.float32)[valid_targets],
        rtol=2e-2,
        atol=2e-2,
    )


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


def test_ki_compute_loss_forwards_prefix_once(monkeypatch):
    model, observation, actions = _make_ki_model_and_batch()
    call_count = 0
    original = _pi0.Pi0._forward_prefix  # noqa: SLF001

    def counted_forward_prefix(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_pi0.Pi0, "_forward_prefix", counted_forward_prefix)
    total, _ = model.compute_loss_with_aux(jax.random.key(3), observation, actions, train=False)

    assert np.isfinite(total)
    assert call_count == 1
