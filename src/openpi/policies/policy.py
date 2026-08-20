from collections.abc import Mapping, Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import rtc_guidance
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._supports_rtc = not is_pytorch and isinstance(model, _pi0.Pi0)

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = (
                nnx_utils.module_jit(model.sample_actions, static_argnames=("rtc_use_vjp",))
                if self._supports_rtc
                else nnx_utils.module_jit(model.sample_actions)
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc: Mapping[str, Any] | None = None,
    ) -> dict:  # type: ignore[misc]
        """Run policy inference, optionally with basic test-time RTC guidance.

        ``rtc["prev_actions_abs"]`` must contain the remaining absolute action
        trajectory aligned to the current observation time. It is intentionally
        passed through the normal input transforms so robot-specific relative
        action conversion, checkpoint normalization, and model padding stay
        identical to the training representation.
        """
        if rtc is not None and not isinstance(rtc, Mapping):
            raise TypeError(f"rtc must be a mapping or None, got {type(rtc).__name__}")
        if rtc is not None and not self._supports_rtc:
            raise ValueError("RTC is only supported by JAX Pi0/Pi0.5 policies")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if rtc is not None and (prev_actions_abs := rtc.get("prev_actions_abs")) is not None:
            prev_actions_abs = np.asarray(prev_actions_abs)
            if prev_actions_abs.ndim != 2:
                raise ValueError(
                    f"rtc.prev_actions_abs must have shape [steps, physical_action_dim], got {prev_actions_abs.shape}"
                )
            if not np.all(np.isfinite(prev_actions_abs)):
                raise ValueError("rtc.prev_actions_abs must contain only finite values")
            # Some transforms (including DeltaActions) update actions in place.
            # Copy here so inference never mutates the client's previous chunk.
            inputs["actions"] = np.array(prev_actions_abs, copy=True)
        inputs = self._input_transform(inputs)

        rtc_info = None
        if self._supports_rtc:
            (
                rtc_prev_actions,
                rtc_prefix_weights,
                rtc_max_guidance_weight,
                rtc_use_vjp,
                rtc_info,
            ) = self._prepare_rtc_inputs(inputs, rtc)  # rtc_prev_actions: 固定action_horizon长度的相对动作序列

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if self._supports_rtc:
            sample_kwargs.update(
                rtc_prev_actions=jnp.asarray(rtc_prev_actions)[None, ...],
                rtc_prefix_weights=jnp.asarray(rtc_prefix_weights),
                rtc_max_guidance_weight=jnp.asarray(rtc_max_guidance_weight, dtype=jnp.float32),
                rtc_use_vjp=rtc_use_vjp,
            )
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        if rtc is not None:
            outputs["rtc_info"] = rtc_info
        return outputs

    def _prepare_rtc_inputs(
        self,
        inputs: dict,
        rtc: Mapping[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.float32, bool, dict[str, Any]]:
        """Convert transformed RTC actions to fixed-shape sampler inputs."""
        action_horizon = self._model.action_horizon
        action_dim = self._model.action_dim
        # 变换到 relative format 下, 且经过 normalization 等操作.
        prev_actions = inputs.pop("actions", None) if rtc is not None else None
        if rtc is not None and rtc.get("prev_actions_abs") is not None and prev_actions is None:
            raise ValueError(
                "The policy input transforms dropped rtc.prev_actions_abs; robot input transforms must pass "
                "actions through to the existing relative-action, normalization, and padding transforms"
            )

        if rtc is None:
            prefix_len = 0
            decay_end = 0
            schedule = "exp"
            max_guidance_weight = 0.0
            use_vjp_requested = False
        else:
            prefix_len = int(rtc.get("prefix_len", 0))
            decay_end = int(rtc.get("decay_end", min(2 * prefix_len, action_horizon)))
            schedule = str(rtc.get("schedule", "exp"))
            max_guidance_weight = float(rtc.get("max_guidance_weight", 5.0))    # 对应 VJP 中的 beta
            use_vjp_value = rtc.get("use_vjp", False)
            if not isinstance(use_vjp_value, (bool, np.bool_)):
                raise TypeError(f"rtc.use_vjp must be a boolean, got {type(use_vjp_value).__name__}")
            use_vjp_requested = bool(use_vjp_value)

        if not np.isfinite(max_guidance_weight) or max_guidance_weight < 0:
            raise ValueError(f"rtc.max_guidance_weight must be finite and non-negative, got {max_guidance_weight}")

        prefix_weights = rtc_guidance.compute_prefix_weights(
            action_horizon,
            prefix_len=prefix_len,
            decay_end=decay_end,
            schedule=schedule,
        )
        fixed_prev_actions = np.zeros((action_horizon, action_dim), dtype=np.float32)
        prev_action_steps = 0
        if prev_actions is not None:
            prev_actions = np.asarray(prev_actions)
            if prev_actions.ndim != 2 or prev_actions.shape[1] != action_dim:
                raise ValueError(
                    "The input transform must convert rtc.prev_actions_abs to "
                    f"[steps, {action_dim}] model action space; got {prev_actions.shape}"
                )
            prev_action_steps = prev_actions.shape[0]
            if prev_action_steps > action_horizon:
                raise ValueError(
                    f"RTC reference has {prev_action_steps} transformed steps, exceeding action_horizon={action_horizon}"
                )
            fixed_prev_actions[:prev_action_steps] = prev_actions

        valid_mask = np.arange(action_horizon) < prev_action_steps
        # prefix_weights 中超过 prev_action_steps 的部分置为 0 (因为没有对应的 prev_actions).
        prefix_weights *= valid_mask.astype(np.float32)
        enabled = bool(prev_action_steps and max_guidance_weight > 0 and np.any(prefix_weights))
        use_vjp = enabled and use_vjp_requested
        info = {
            "enabled": enabled,
            "use_vjp": use_vjp,
            "prefix_len": prefix_len,
            "decay_end": decay_end,
            "max_guidance_weight": max_guidance_weight,
            "prev_action_steps": prev_action_steps,
        }
        return fixed_prev_actions, prefix_weights, np.float32(max_guidance_weight), use_vjp, info

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, *, rtc: Mapping[str, Any] | None = None) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs) if rtc is None else self._policy.infer(obs, rtc=rtc)

        data = {"inputs": obs, "outputs": results}
        if rtc is not None:
            data["rtc"] = rtc
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
