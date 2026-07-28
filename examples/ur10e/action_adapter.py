from collections.abc import Mapping, Sequence
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
    def infer(self, obs: dict) -> dict:
        state = np.asarray(obs["observation.state"]).copy()
        result = self._policy.infer(obs)
        if "actions" not in result:
            raise ValueError("Policy result does not contain actions")
        actions = np.asarray(result["actions"])
        if actions.shape != (self._prediction_horizon, UR_ACTION_DIM):
            raise ValueError(
                "Expected policy actions with shape "
                f"({self._prediction_horizon}, {UR_ACTION_DIM}), got {actions.shape}"
            )
        return {**result, "actions": relative_actions_to_absolute(state, actions)}

    @override
    def reset(self) -> None:
        self._policy.reset()


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
        raise ValueError(
            f"execution_horizon must be between 1 and {prediction_horizon}, got {execution_horizon}"
        )

    return action_chunk_broker.ActionChunkBroker(
        policy=RelativeTCPToAbsolutePolicy(policy, prediction_horizon),
        # ActionChunkBroker uses this value as the number of cached actions to execute.
        action_horizon=execution_horizon,
    )
