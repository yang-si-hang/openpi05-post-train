from collections.abc import Mapping, Sequence
import concurrent.futures
import copy
from typing import Any

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import base_policy as _base_policy
from scipy.spatial.transform import Rotation
from typing_extensions import override

UR_ACTION_DIM = 10


def rotation_to_rot6d(rotation: Rotation) -> np.ndarray:
    """Encode a single rotation using the first two columns of its matrix."""

    matrix = rotation.as_matrix()
    if matrix.shape != (3, 3):
        raise ValueError("Expected a single rotation")
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def rot6d_to_rotation(rot6d: Sequence[float]) -> Rotation:
    """Decode a 6D rotation with Gram-Schmidt orthonormalization."""

    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Rot6D must contain 6 values, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("Rot6D values must be finite")

    first = values[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("Rot6D first direction must be non-zero")
    first = first / first_norm

    second = values[3:] - np.dot(first, values[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("Rot6D directions must not be parallel")
    second = second / second_norm

    third = np.cross(first, second)
    return Rotation.from_matrix(np.column_stack((first, second, third)))


def relative_actions_to_absolute(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert a TCP-frame relative action chunk to base-frame absolute TCP targets."""

    state_values = np.asarray(state, dtype=np.float64)
    action_array = np.asarray(actions)
    action_values = np.asarray(actions, dtype=np.float64)
    if state_values.shape != (UR_ACTION_DIM,):
        raise ValueError(f"UR state must have shape ({UR_ACTION_DIM},), got {state_values.shape}")
    if action_values.ndim == 0 or action_values.shape[-1] != UR_ACTION_DIM:
        raise ValueError(f"UR actions must have last dimension {UR_ACTION_DIM}, got {action_values.shape}")
    if not np.all(np.isfinite(state_values)) or not np.all(np.isfinite(action_values)):
        raise ValueError("UR state and action values must be finite")

    current_position = state_values[:3]
    current_rotation = rot6d_to_rotation(state_values[3:9])
    absolute_actions = action_values.copy()
    flat_actions = action_values.reshape(-1, UR_ACTION_DIM)
    flat_absolute_actions = absolute_actions.reshape(-1, UR_ACTION_DIM)
    for index, relative in enumerate(flat_actions):
        relative_rotation = rot6d_to_rotation(relative[3:9])
        flat_absolute_actions[index, :3] = current_position + current_rotation.apply(relative[:3])
        flat_absolute_actions[index, 3:9] = rotation_to_rot6d(current_rotation * relative_rotation)

    output_dtype = action_array.dtype if np.issubdtype(action_array.dtype, np.floating) else np.dtype(np.float32)
    return absolute_actions.astype(output_dtype, copy=False)


class RelativeTCPToAbsolutePolicy(_base_policy.BasePolicy):
    """Convert a complete relative policy chunk before it reaches the action broker."""

    def __init__(self, policy: _base_policy.BasePolicy, prediction_horizon: int):
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        self._policy = policy
        self._prediction_horizon = prediction_horizon

    @override
    def infer(self, obs: dict, *, rtc: Mapping[str, Any] | None = None) -> dict:
        state = np.asarray(obs["observation.state"]).copy()
        result = self._policy.infer(obs) if rtc is None else self._policy.infer(obs, rtc=rtc)
        if "actions" not in result:
            raise ValueError("Policy result does not contain actions")
        actions = np.asarray(result["actions"])
        if actions.shape != (self._prediction_horizon, UR_ACTION_DIM):
            raise ValueError(
                f"Expected policy actions with shape ({self._prediction_horizon}, {UR_ACTION_DIM}), got {actions.shape}"
            )
        return {**result, "actions": relative_actions_to_absolute(state, actions)}

    @override
    def reset(self) -> None:
        self._policy.reset()


class RTCActionChunkBroker(_base_policy.BasePolicy):
    """Prefetch RTC-guided chunks while continuing to execute the old absolute chunk.

    ``infer`` is called once per policy timestep and returns one absolute action.
    Initial inference is synchronous. Once ``replan_interval`` actions from a
    chunk have been consumed, the next inference runs on a background thread;
    calls made while it is pending continue to consume the old chunk.

    The wrapped policy must return a complete absolute action chunk and accept
    ``infer(obs, rtc=...)``. ``RelativeTCPToAbsolutePolicy`` provides exactly
    that contract for the UR WebSocket policy.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        prediction_horizon: int,
        replan_interval: int,
        prefix_len: int,
        decay_end: int | None = None,
        schedule: str = "exp",
        max_guidance_weight: float = 5.0,
        use_vjp: bool = False,
    ):
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if not 0 < replan_interval < prediction_horizon:
            raise ValueError("replan_interval must be between 1 and prediction_horizon - 1")
        remaining_horizon = prediction_horizon - replan_interval
        if not 0 < prefix_len <= remaining_horizon:
            raise ValueError(f"prefix_len must be between 1 and {remaining_horizon}")
        if decay_end is None:
            decay_end = min(2 * prefix_len, remaining_horizon)
        if not prefix_len <= decay_end <= prediction_horizon:
            raise ValueError(f"decay_end must be between prefix_len={prefix_len} and {prediction_horizon}")
        if schedule not in {"exp", "linear", "ones", "zeros"}:
            raise ValueError(f"Unknown RTC schedule: {schedule}")
        if not np.isfinite(max_guidance_weight) or max_guidance_weight < 0:
            raise ValueError("max_guidance_weight must be finite and non-negative")
        if not isinstance(use_vjp, (bool, np.bool_)):
            raise TypeError("use_vjp must be a boolean")

        self._policy = policy
        self._prediction_horizon = prediction_horizon
        self._replan_interval = replan_interval
        self._prefix_len = prefix_len
        self._decay_end = decay_end
        self._schedule = schedule
        self._max_guidance_weight = max_guidance_weight
        self._use_vjp = bool(use_vjp)

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-rtc")
        self._last_results: dict[str, Any] | None = None
        self._cur_step = 0
        self._pending_future: concurrent.futures.Future[dict] | None = None
        self._request_cursor: int | None = None
        self._last_replan_elapsed_steps: int | None = None

    @override
    def infer(self, obs: dict) -> dict:
        if self._last_results is None:
            self._install_initial_chunk(self._policy.infer(obs))
        else:
            self._maybe_install_pending_chunk()

        if self._pending_future is None and self._cur_step >= self._replan_interval:
            self._start_async_inference(obs)

        # If inference has not completed before the old prediction horizon is
        # exhausted, wait for it rather than indexing beyond the old chunk.
        if self._cur_step >= self._prediction_horizon:
            self._install_pending_chunk(block=True)

        result = self._slice_result(self._last_results, self._cur_step)
        result["rtc_broker_info"] = {
            "chunk_step": self._cur_step,
            "inference_pending": self._pending_future is not None,
            "last_replan_elapsed_steps": self._last_replan_elapsed_steps,
        }
        self._cur_step += 1
        return result

    def _install_initial_chunk(self, result: dict) -> None:
        self._validate_chunk(result)
        self._last_results = result
        self._cur_step = 0
        self._last_replan_elapsed_steps = None

    def _start_async_inference(self, obs: dict) -> None:
        assert self._last_results is not None
        actions = np.asarray(self._last_results["actions"])
        request_cursor = self._cur_step
        prev_actions_abs = actions[request_cursor:].copy()
        if len(prev_actions_abs) == 0:
            raise RuntimeError("Cannot start RTC inference without remaining actions")

        rtc = {
            "prev_actions_abs": prev_actions_abs,
            "prefix_len": self._prefix_len,
            "decay_end": self._decay_end,
            "schedule": self._schedule,
            "max_guidance_weight": self._max_guidance_weight,
            "use_vjp": self._use_vjp,
        }
        # Snapshot the observation at exactly the same cursor used to slice the
        # previous chunk. The caller may mutate its observation after this call.
        observation = copy.deepcopy(obs)
        self._request_cursor = request_cursor
        self._pending_future = self._executor.submit(self._policy.infer, observation, rtc=rtc)

    def _maybe_install_pending_chunk(self) -> None:
        if self._pending_future is None or self._request_cursor is None:
            return
        elapsed_steps = self._cur_step - self._request_cursor
        # Even if inference returns early, execute the committed prefix before
        # switching so the model's prefix_len contract remains true.
        if self._pending_future.done() and elapsed_steps >= self._prefix_len:
            self._install_pending_chunk(block=False)

    def _install_pending_chunk(self, *, block: bool) -> None:
        if self._pending_future is None or self._request_cursor is None:
            raise RuntimeError("No RTC inference is pending")
        if not block and not self._pending_future.done():
            return

        result = self._pending_future.result()
        self._validate_chunk(result)
        elapsed_steps = self._cur_step - self._request_cursor
        if not 0 <= elapsed_steps < self._prediction_horizon:
            raise RuntimeError(
                f"New RTC chunk is already exhausted: elapsed_steps={elapsed_steps}, "
                f"prediction_horizon={self._prediction_horizon}"
            )

        self._last_results = result
        # N0 ... N(elapsed_steps-1) correspond to old actions that were
        # executed while inference was running, so resume at N(elapsed_steps).
        self._cur_step = elapsed_steps
        self._last_replan_elapsed_steps = elapsed_steps
        self._pending_future = None
        self._request_cursor = None

    def _validate_chunk(self, result: dict) -> None:
        if "actions" not in result:
            raise ValueError("Policy result does not contain actions")
        actions = np.asarray(result["actions"])
        if actions.ndim != 2 or actions.shape[0] != self._prediction_horizon:
            raise ValueError(
                f"Expected policy actions with shape ({self._prediction_horizon}, action_dim), got {actions.shape}"
            )

    @staticmethod
    def _slice_result(result: dict[str, Any], step: int) -> dict[str, Any]:
        return {
            key: value[step, ...] if isinstance(value, np.ndarray) and value.ndim > 0 else value
            for key, value in result.items()
        }

    @override
    def reset(self) -> None:
        # A running WebSocket request must be drained before reusing the same
        # synchronous connection for a new episode.
        try:
            if self._pending_future is not None:
                self._pending_future.result()
        finally:
            self._pending_future = None
            self._request_cursor = None
            self._last_results = None
            self._cur_step = 0
            self._last_replan_elapsed_steps = None
            self._policy.reset()

    def close(self) -> None:
        """Drain any request and stop the background inference worker."""
        try:
            self.reset()
        finally:
            self._executor.shutdown(wait=True)


def create_absolute_action_broker(
    policy: _base_policy.BasePolicy,
    metadata: Mapping[str, Any],
    *,
    execution_horizon: int = 10,
) -> action_chunk_broker.ActionChunkBroker:
    """Build a broker that executes absolute actions and replans at the requested horizon."""

    prediction_horizon = int(metadata.get("prediction_horizon", 0))
    action_dim = int(metadata.get("action_dim", 0))
    if prediction_horizon <= 0:
        raise ValueError("Server metadata must contain a positive prediction_horizon")
    if action_dim != UR_ACTION_DIM:
        raise ValueError(f"Server metadata action_dim must be {UR_ACTION_DIM}, got {action_dim}")
    if execution_horizon <= 0 or execution_horizon > prediction_horizon:
        raise ValueError(f"execution_horizon must be between 1 and {prediction_horizon}, got {execution_horizon}")

    return action_chunk_broker.ActionChunkBroker(
        policy=RelativeTCPToAbsolutePolicy(policy, prediction_horizon),
        # ActionChunkBroker uses this value as the number of cached actions to execute.
        action_horizon=execution_horizon,
    )


def create_rtc_action_broker(
    policy: _base_policy.BasePolicy,
    metadata: Mapping[str, Any],
    *,
    replan_interval: int = 10,
    prefix_len: int = 1,
    decay_end: int | None = None,
    schedule: str = "exp",
    max_guidance_weight: float = 5.0,
    use_vjp: bool = False,
) -> RTCActionChunkBroker:
    """Build an asynchronous absolute-action broker for test-time RTC."""
    prediction_horizon = int(metadata.get("prediction_horizon", 0))
    action_dim = int(metadata.get("action_dim", 0))
    if prediction_horizon <= 0:
        raise ValueError("Server metadata must contain a positive prediction_horizon")
    if action_dim != UR_ACTION_DIM:
        raise ValueError(f"Server metadata action_dim must be {UR_ACTION_DIM}, got {action_dim}")

    return RTCActionChunkBroker(
        policy=RelativeTCPToAbsolutePolicy(policy, prediction_horizon),
        prediction_horizon=prediction_horizon,
        replan_interval=replan_interval,
        prefix_len=prefix_len,
        decay_end=decay_end,
        schedule=schedule,
        max_guidance_weight=max_guidance_weight,
        use_vjp=use_vjp,
    )
