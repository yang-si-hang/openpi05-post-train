import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc_guidance
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def sample_train_time_rtc_delay(rng, batch_shape: tuple[int, ...], max_delay: int):
    """Sample delays uniformly from the closed integer interval [0, max_delay]."""
    return jax.random.randint(rng, batch_shape, minval=0, maxval=max_delay + 1)


def apply_train_time_rtc_corruption(actions, noise, time, delay):
    """Keep the delayed prefix clean and apply flow corruption to the postfix."""
    prefix_mask = jnp.arange(actions.shape[-2]) < delay[..., None]
    token_time = jnp.where(prefix_mask, 0.0, time[..., None])
    x_t = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
    return x_t, token_time, prefix_mask


def normalize_postfix_loss(loss, prefix_mask):
    """Keep a [B, H] loss while making its outer mean a valid-token mean."""
    postfix_mask = jnp.logical_not(prefix_mask)
    masked_loss = loss * postfix_mask
    return masked_loss * (masked_loss.size / jnp.sum(postfix_mask))


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.train_time_rtc_max_delay = config.train_time_rtc_max_delay
        self.knowledge_insulation = config.knowledge_insulation
        self.ki_fast_loss_weight = config.ki_fast_loss_weight
        self.ki_flow_loss_weight = config.ki_flow_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    def embed_ki_inputs(self, obs: _model.Observation):
        """Embed images and fixed-length FAST supervision tokens."""
        if any(x is None for x in (obs.ki_tokens, obs.ki_token_mask, obs.ki_ar_mask, obs.ki_loss_mask)):
            raise ValueError("KI observation fields must all be present")
        tokens, masks, ar_masks = [], [], []
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            tokens.append(image_tokens)
            masks.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_masks.append(jnp.zeros(image_tokens.shape[:2], dtype=jnp.bool_))
        tokens.append(self.PaliGemma.llm(obs.ki_tokens, method="embed"))
        masks.append(obs.ki_token_mask)
        ar_masks.append(obs.ki_ar_mask)
        return jnp.concatenate(tokens, axis=1), jnp.concatenate(masks, axis=1), jnp.concatenate(ar_masks, axis=1)

    def _compute_fast_loss(self, observation: _model.Observation):
        full_embeddings, full_input_mask, full_ar_mask = self.embed_ki_inputs(observation)
        full_attn_mask = make_attn_mask(full_input_mask, full_ar_mask)
        full_positions = jnp.cumsum(full_input_mask, axis=1) - 1
        model_embeddings = full_embeddings[:, :-1]
        model_attn_mask = full_attn_mask[:, :-1, :-1]
        model_positions = full_positions[:, :-1]
        assert model_embeddings.shape[1] == full_embeddings.shape[1] - 1
        assert model_attn_mask.shape[-2:] == (model_embeddings.shape[1], model_embeddings.shape[1])
        assert model_positions.shape[1] == model_embeddings.shape[1]
        (vlm_out, _), _ = self.PaliGemma.llm([model_embeddings, None], mask=model_attn_mask, positions=model_positions)
        targets = observation.ki_tokens[:, 1:]
        loss_mask = observation.ki_loss_mask[:, 1:]
        token_hidden = vlm_out[:, -targets.shape[1] :]
        assert token_hidden.shape[:2] == targets.shape
        assert targets.shape == loss_mask.shape
        logits = self.PaliGemma.llm(token_hidden, method="decode")
        token_ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]
        counts = jnp.sum(loss_mask, axis=-1)
        per_example = jnp.sum(token_ce * loss_mask, axis=-1) / jnp.maximum(counts, 1)
        correct = jnp.sum((jnp.argmax(logits, axis=-1) == targets) * loss_mask)
        return per_example, correct, jnp.sum(counts)

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        if timestep.ndim == 1:
            time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        elif timestep.ndim == 2:
            time_emb = jax.vmap(
                lambda token_time: posemb_sincos(
                    token_time,
                    self.action_in_proj.out_features,
                    min_period=4e-3,
                    max_period=4.0,
                )
            )(timestep)
        else:
            raise ValueError(f"timestep must have shape [batch] or [batch, action_horizon], got {timestep.shape}")
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = (
                einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon) if time_emb.ndim == 2 else time_emb
            )
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.train_time_rtc_max_delay > 0:
            preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        else:
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        if self.train_time_rtc_max_delay > 0:
            delay = sample_train_time_rtc_delay(delay_rng, batch_shape, self.train_time_rtc_max_delay)
            x_t, time, action_prefix_mask = apply_train_time_rtc_corruption(actions, noise, time, delay)
        else:
            action_prefix_mask = None
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions  # velocity ground truth

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if action_prefix_mask is None:
            return loss

        return normalize_postfix_loss(loss, action_prefix_mask)

    def _compute_flow_loss_preprocessed(self, noise_rng, time_rng, delay_rng, observation, actions):
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        if self.train_time_rtc_max_delay > 0:
            delay = sample_train_time_rtc_delay(delay_rng, batch_shape, self.train_time_rtc_max_delay)
            x_t, time, action_prefix_mask = apply_train_time_rtc_corruption(actions, noise, time, delay)
        else:
            action_prefix_mask = None
            x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_region = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_region, suffix_attn_mask], axis=-1)
        suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            kv_cache=kv_cache,
            mask=full_attn_mask,
            positions=suffix_positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return loss if action_prefix_mask is None else normalize_postfix_loss(loss, action_prefix_mask)

    def compute_loss_with_aux(self, rng, observation, actions, *, train=False):
        """Compute scalar KI loss and count-based training metrics."""
        if not self.knowledge_insulation:
            flow = self.compute_loss(rng, observation, actions, train=train)
            scalar = jnp.mean(flow)
            zero = jnp.asarray(0.0)
            return scalar, {
                "flow_loss": scalar,
                "fast_ce_loss": zero,
                "fast_correct_count": zero,
                "fast_target_token_count": zero,
            }
        if self.train_time_rtc_max_delay > 0:
            preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        else:
            preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
            delay_rng = None
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        fast, correct, target_count = self._compute_fast_loss(observation)
        flow = self._compute_flow_loss_preprocessed(noise_rng, time_rng, delay_rng, observation, actions)
        fast_scalar, flow_scalar = jnp.mean(fast), jnp.mean(flow)
        total = self.ki_fast_loss_weight * fast_scalar + self.ki_flow_loss_weight * flow_scalar
        return total, {
            "flow_loss": flow_scalar,
            "fast_ce_loss": fast_scalar,
            "fast_correct_count": correct,
            "fast_target_token_count": target_count,
        }

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        rtc_prev_actions: at.Float[at.Array, "b ah ad"] | None = None,
        rtc_prefix_len: at.Int[at.Array, " b"] | None = None,
        rtc_prefix_weights: at.Float[at.Array, " ah"] | None = None,
        rtc_max_guidance_weight: float | at.Float[at.Array, ""] = 0.0,
        rtc_use_vjp: bool = False,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # Keep direct callers backward compatible. Policy supplies these fixed-shape
        # zero tensors even when RTC is disabled to keep its JIT signature stable.
        if rtc_prev_actions is None:
            rtc_prev_actions = jnp.zeros((batch_size, self.action_horizon, self.action_dim), dtype=noise.dtype)
        if rtc_prefix_len is None:
            rtc_prefix_len = jnp.zeros((batch_size,), dtype=jnp.int32)
        if rtc_prefix_weights is None:
            rtc_prefix_weights = jnp.zeros((self.action_horizon,), dtype=noise.dtype)

        hard_prefix_mask = jnp.arange(self.action_horizon)[None, :] < rtc_prefix_len[:, None]

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def denoise(x_t, time, *, hard_prefix: bool = False):
            denoise_time = jnp.broadcast_to(time, (batch_size,))
            if hard_prefix:
                denoise_time = jnp.where(hard_prefix_mask, 0.0, denoise_time[:, None])
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, denoise_time)
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step(carry):
            x_t, time = carry
            if self.train_time_rtc_max_delay > 0:
                x_t = jnp.where(hard_prefix_mask[..., None], rtc_prev_actions, x_t)
                v_t = denoise(x_t, time, hard_prefix=True)
                # Euler integration applies only to the unconditioned postfix.
                x_t = jnp.where(hard_prefix_mask[..., None], x_t, x_t + dt * v_t)
                return x_t, time + dt
            if rtc_use_vjp:
                v_t = rtc_guidance.apply_vjp_rtc_guidance(
                    x_t=x_t,
                    denoise_fn=lambda x: denoise(x, time),
                    prev_actions=rtc_prev_actions,
                    prefix_weights=rtc_prefix_weights,
                    openpi_time=time,
                    max_guidance_weight=rtc_max_guidance_weight,
                )
            else:
                v_t = denoise(x_t, time)
                v_t = rtc_guidance.apply_basic_rtc_guidance(
                    x_t=x_t,
                    v_t=v_t,
                    prev_actions=rtc_prev_actions,
                    prefix_weights=rtc_prefix_weights,
                    openpi_time=time,
                    max_guidance_weight=rtc_max_guidance_weight,
                )

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        if self.train_time_rtc_max_delay > 0:
            x_0 = jnp.where(hard_prefix_mask[..., None], rtc_prev_actions, x_0)
        return x_0
