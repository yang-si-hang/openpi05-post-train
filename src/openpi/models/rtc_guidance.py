"""Inference-only Real-Time Chunking (RTC) guidance for JAX Pi0 models."""

import jax
import jax.numpy as jnp
import numpy as np


def compute_prefix_weights(
    action_horizon: int,
    prefix_len: int,
    decay_end: int,
    schedule: str = "exp",
) -> np.ndarray:
    """Build fixed-horizon per-step weights for an RTC reference trajectory.

    ``prefix_len == 0`` disables RTC and therefore returns all-zero weights,
    matching the test-time RTC behavior used by FluxVLA.

    For the default ``exp`` schedule this implements Eq. (5) from
    "Real-Time Execution of Action Chunking Flow Policies". In the paper's
    notation, ``action_horizon`` is H, ``prefix_len`` is d, and ``decay_end``
    is H-s. Thus, for ``prefix_len <= i < decay_end``:

        c_i = (decay_end - i) / (decay_end - prefix_len + 1)
        W_i = c_i * (exp(c_i) - 1) / (e - 1)

    The other schedules are retained as ablations over the same soft-mask
    interval and are not the paper's Eq. (5).
    """
    if action_horizon < 0:
        raise ValueError(f"action_horizon must be non-negative, got {action_horizon}")
    if not 0 <= prefix_len <= action_horizon:
        raise ValueError(f"prefix_len must be in [0, {action_horizon}], got {prefix_len}")
    if not prefix_len <= decay_end <= action_horizon:
        raise ValueError(f"decay_end must be in [{prefix_len}, {action_horizon}], got {decay_end}")
    if schedule not in {"exp", "linear", "ones", "zeros"}:
        raise ValueError(f"Unknown RTC prefix weight schedule: {schedule}")

    weights = np.zeros(action_horizon, dtype=np.float32)
    if prefix_len == 0:
        return weights

    weights[:prefix_len] = 1.0
    soft_mask_len = decay_end - prefix_len
    for position in range(prefix_len, decay_end):
        c_i = (decay_end - position) / (soft_mask_len + 1)
        if schedule == "exp":
            weights[position] = c_i * np.expm1(c_i) / np.expm1(1.0)
        elif schedule == "linear":
            weights[position] = c_i
        elif schedule == "ones":
            weights[position] = 1.0
        # The array is initialized to zero, so the "zeros" schedule needs no assignment.

    return weights


def compute_guidance_weight(rtc_time, max_guidance_weight):
    """Compute the non-VJP RTC scale for ``rtc_time`` (0=noise, 1=clean).

    This function only uses JAX operations so it is safe inside a JIT-compiled
    denoising loop.
    """
    eps = 1e-6
    time = jnp.maximum(rtc_time, eps)
    one_minus_time = jnp.maximum(1.0 - rtc_time, eps)

    inv_r2 = (time**2 + one_minus_time**2) / (one_minus_time**2)
    c = one_minus_time / time
    return jnp.minimum(c * inv_r2, max_guidance_weight)


def apply_basic_rtc_guidance(
    x_t,
    v_t,
    prev_actions,
    prefix_weights,
    openpi_time,
    max_guidance_weight,
):
    """Apply basic test-time RTC guidance without a VJP or extra denoiser call.

    OpenPI flows from ``t=1`` noise to ``t=0`` clean, whereas the RTC/FluxVLA
    convention flows from 0 noise to 1 clean. In OpenPI's convention the clean
    endpoint estimate is ``a_hat = x_t - t * v_t``. FluxVLA negates the Pi0
    velocity before guidance and negates it again afterwards; expressed directly
    in OpenPI's convention, that gives ``v_guided = v_t - gw * error``.
    """
    rtc_time = 1.0 - openpi_time
    action_estimate = x_t - openpi_time * v_t
    weights = prefix_weights[None, :, None]
    error = (prev_actions - action_estimate) * weights
    guidance_weight = compute_guidance_weight(rtc_time, max_guidance_weight)
    return v_t - guidance_weight * error


def apply_vjp_rtc_guidance(
    x_t,
    denoise_fn,
    prev_actions,
    prefix_weights,
    openpi_time,
    max_guidance_weight,
):
    """Apply RTC guidance using the endpoint Jacobian transpose.

    ``denoise_fn`` returns velocity in OpenPI's convention (t=1 noise to
    t=0 clean). The endpoint map is therefore ``a_hat(x) = x - t*v(x)``.
    ``jax.vjp`` evaluates the denoiser once in its forward pass and returns
    ``J_a_hat.T @ error`` from the pullback. Converting the FluxVLA/RTC
    velocity back to OpenPI's convention gives the final negative correction.
    """

    def endpoint_fn(x):
        velocity = denoise_fn(x)
        action_estimate = x - openpi_time * velocity
        return action_estimate, velocity

    action_estimate, pullback, v_t = jax.vjp(endpoint_fn, x_t, has_aux=True)
    weights = prefix_weights[None, :, None]  # add batch and action dims
    error = (prev_actions - action_estimate) * weights
    correction = pullback(error)[0]  # pullback is back gradient
    guidance_weight = compute_guidance_weight(1.0 - openpi_time, max_guidance_weight)
    return v_t - guidance_weight * correction
