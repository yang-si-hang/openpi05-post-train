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
        self._supports_train_time_rtc = bool(self._supports_rtc and getattr(model, "train_time_rtc_max_delay", 0) > 0)

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
        """Run policy inference with an explicitly selected RTC mode.

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
        rtc_mode = self._resolve_rtc_mode(rtc)

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        if rtc_mode != "off" and (prev_actions_abs := rtc.get("prev_actions_abs")) is not None:
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
                rtc_prefix_len,
            ) = self._prepare_rtc_inputs(inputs, rtc, rtc_mode)

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
                rtc_prefix_len=jnp.asarray([rtc_prefix_len], dtype=jnp.int32),
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
        if rtc_mode != "off":
            outputs["rtc_info"] = rtc_info
        return outputs

    def _resolve_rtc_mode(self, rtc: Mapping[str, Any] | None) -> str:
        """Validate the request mode against the checkpoint configuration."""
        if rtc is None:
            return "off"
        mode_value = rtc.get("mode", "off")
        if not isinstance(mode_value, str):
            raise TypeError(f"rtc.mode must be a string, got {type(mode_value).__name__}")
        mode = mode_value.lower()
        if mode not in {"off", "non_vjp", "vjp", "train_rtc"}:
            raise ValueError(f"Unknown rtc.mode: {mode_value}")
        if mode == "off":
            extra_fields = set(rtc) - {"mode"}
            if extra_fields:
                raise ValueError(f"RTC parameters are not allowed when rtc.mode is off: {sorted(extra_fields)}")
            return mode
        if self._supports_train_time_rtc and mode != "train_rtc":
            raise ValueError("inference-time RTC guidance cannot be used with a training-time RTC checkpoint")
        if not self._supports_train_time_rtc and mode == "train_rtc":
            raise ValueError("train_rtc requires a checkpoint configured with training-time action conditioning")
        if mode in {"non_vjp", "vjp"} and "use_vjp" in rtc:
            raise ValueError("rtc.use_vjp is not allowed when rtc.mode explicitly selects the guidance method")
        if mode == "train_rtc":
            forbidden = {"decay_end", "schedule", "max_guidance_weight", "use_vjp"}.intersection(rtc)
            if forbidden:
                raise ValueError(f"train_rtc does not support inference-time guidance fields: {sorted(forbidden)}")
        allowed_fields = {"mode", "prev_actions_abs", "prefix_len"}
        if mode in {"non_vjp", "vjp"}:
            allowed_fields.update({"decay_end", "schedule", "max_guidance_weight"})
        unknown_fields = set(rtc) - allowed_fields
        if unknown_fields:
            raise ValueError(f"Unknown RTC request fields: {sorted(unknown_fields)}")
        return mode

    def _prepare_rtc_inputs(
        self,
        inputs: dict,
        rtc: Mapping[str, Any] | None,
        rtc_mode: str,
    ) -> tuple[np.ndarray, np.ndarray, np.float32, bool, dict[str, Any] | None, np.int32]:
        """Convert transformed RTC actions to fixed-shape sampler inputs."""
        action_horizon = self._model.action_horizon
        action_dim = self._model.action_dim
        # 变换到 relative format 下, 且经过 normalization 等操作.
        prev_actions = inputs.pop("actions", None) if rtc_mode != "off" else None
        if rtc_mode != "off" and rtc.get("prev_actions_abs") is not None and prev_actions is None:
            raise ValueError(
                "The policy input transforms dropped rtc.prev_actions_abs; robot input transforms must pass "
                "actions through to the existing relative-action, normalization, and padding transforms"
            )

        if rtc_mode == "off":
            prefix_len = 0
            decay_end = 0
            schedule = "exp"
            max_guidance_weight = 0.0
            use_vjp = False
        else:
            prefix_len_value = rtc.get("prefix_len")
            if isinstance(prefix_len_value, (bool, np.bool_)) or not isinstance(prefix_len_value, (int, np.integer)):
                raise TypeError("rtc.prefix_len must be an integer")
            prefix_len = int(prefix_len_value)
            if prefix_len <= 0:
                raise ValueError("rtc.prefix_len must be positive")
            if prefix_len > action_horizon:
                raise ValueError(f"rtc.prefix_len must not exceed action_horizon={action_horizon}")
            if rtc_mode == "train_rtc":
                max_delay = int(self._model.train_time_rtc_max_delay)
                if prefix_len > max_delay:
                    raise ValueError(f"rtc.prefix_len={prefix_len} exceeds the checkpoint max delay {max_delay}")
                decay_end = prefix_len
                schedule = "zeros"
                max_guidance_weight = 0.0
                use_vjp = False
            else:
                decay_end_value = rtc.get("decay_end", min(2 * prefix_len, action_horizon))
                if isinstance(decay_end_value, (bool, np.bool_)) or not isinstance(decay_end_value, (int, np.integer)):
                    raise TypeError("rtc.decay_end must be an integer")
                decay_end = int(decay_end_value)
                schedule = rtc.get("schedule", "exp")
                if not isinstance(schedule, str):
                    raise TypeError("rtc.schedule must be a string")
                max_guidance_weight_value = rtc.get("max_guidance_weight", 5.0)
                if isinstance(max_guidance_weight_value, (bool, np.bool_)):
                    raise TypeError("rtc.max_guidance_weight must be a number, not bool")
                max_guidance_weight = float(max_guidance_weight_value)
                use_vjp = rtc_mode == "vjp"

        if not np.isfinite(max_guidance_weight) or max_guidance_weight < 0:
            raise ValueError(f"rtc.max_guidance_weight must be finite and non-negative, got {max_guidance_weight}")

        prefix_weights = (
            np.zeros(action_horizon, dtype=np.float32)
            if rtc_mode in {"off", "train_rtc"}
            else rtc_guidance.compute_prefix_weights(
                action_horizon,
                prefix_len=prefix_len,
                decay_end=decay_end,
                schedule=schedule,
            )
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

        if rtc_mode != "off" and prev_action_steps == 0:
            raise ValueError("rtc.prev_actions_abs is required when RTC is enabled")
        if rtc_mode != "off" and prefix_len > prev_action_steps:
            raise ValueError(
                f"rtc.prefix_len={prefix_len} exceeds the available previous actions ({prev_action_steps})"
            )

        valid_mask = np.arange(action_horizon) < prev_action_steps
        # prefix_weights 中超过 prev_action_steps 的部分置为 0 (因为没有对应的 prev_actions).
        prefix_weights *= valid_mask.astype(np.float32)
        enabled = rtc_mode != "off"
        info = (
            None
            if rtc_mode == "off"
            else {
                "mode": rtc_mode,
                "enabled": enabled,
                "prefix_len": prefix_len,
                "prev_action_steps": prev_action_steps,
            }
        )
        return (
            fixed_prev_actions,
            prefix_weights,
            np.float32(max_guidance_weight),
            use_vjp,
            info,
            np.int32(prefix_len),
        )

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
