from collections.abc import Sequence
import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

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


def absolute_actions_to_relative(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Express absolute TCP targets relative to the current TCP frame."""

    state_values, action_values, output_dtype = _validate_state_and_actions(state, actions)
    current_position = state_values[:3]
    current_rotation = rot6d_to_rotation(state_values[3:9])
    current_rotation_inverse = current_rotation.inv()

    relative_actions = action_values.copy()
    flat_actions = action_values.reshape(-1, UR_ACTION_DIM)
    flat_relative_actions = relative_actions.reshape(-1, UR_ACTION_DIM)
    for index, target in enumerate(flat_actions):
        target_rotation = rot6d_to_rotation(target[3:9])
        flat_relative_actions[index, :3] = current_rotation_inverse.apply(target[:3] - current_position)
        flat_relative_actions[index, 3:9] = rotation_to_rot6d(current_rotation_inverse * target_rotation)

    return relative_actions.astype(output_dtype, copy=False)


def relative_actions_to_absolute(state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Convert TCP-frame relative actions to absolute TCP targets in the base frame."""

    state_values, action_values, output_dtype = _validate_state_and_actions(state, actions)
    current_position = state_values[:3]
    current_rotation = rot6d_to_rotation(state_values[3:9])

    absolute_actions = action_values.copy()
    flat_actions = action_values.reshape(-1, UR_ACTION_DIM)
    flat_absolute_actions = absolute_actions.reshape(-1, UR_ACTION_DIM)
    for index, relative in enumerate(flat_actions):
        relative_rotation = rot6d_to_rotation(relative[3:9])
        flat_absolute_actions[index, :3] = current_position + current_rotation.apply(relative[:3])
        flat_absolute_actions[index, 3:9] = rotation_to_rot6d(current_rotation * relative_rotation)

    return absolute_actions.astype(output_dtype, copy=False)


def make_ur_example() -> dict:
    """Create an example observation following the UR policy input contract."""

    return {
        "observation.images.base_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.images.left_wrist_0_rgb": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.concatenate(
            [
                np.zeros(3, dtype=np.float32),
                rotation_to_rot6d(Rotation.identity()).astype(np.float32),
                np.zeros(1, dtype=np.float32),
            ]
        ),
        "prompt": "do something",
    }


def _validate_state_and_actions(
    state: np.ndarray, actions: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.dtype]:
    state_values = np.asarray(state, dtype=np.float64)
    action_array = np.asarray(actions)
    action_values = np.asarray(actions, dtype=np.float64)

    if state_values.shape != (UR_ACTION_DIM,):
        raise ValueError(f"UR state must have shape ({UR_ACTION_DIM},), got {state_values.shape}")
    if action_values.ndim == 0 or action_values.shape[-1] != UR_ACTION_DIM:
        raise ValueError(f"UR actions must have last dimension {UR_ACTION_DIM}, got {action_values.shape}")
    if not np.all(np.isfinite(state_values)):
        raise ValueError("UR state values must be finite")
    if not np.all(np.isfinite(action_values)):
        raise ValueError("UR action values must be finite")

    output_dtype = action_array.dtype if np.issubdtype(action_array.dtype, np.floating) else np.dtype(np.float32)
    return state_values, action_values, output_dtype


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"UR images must be rank 3, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"UR images must have three channels, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class URInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Unsupported model type for UR policy: {self.model_type}")

        state = np.asarray(data["observation.state"])
        if state.shape != (UR_ACTION_DIM,):
            raise ValueError(f"UR state must have shape ({UR_ACTION_DIM},), got {state.shape}")
        if not np.all(np.isfinite(state)):
            raise ValueError("UR state values must be finite")

        base_image = _parse_image(data["observation.images.base_0_rgb"])
        wrist_image = _parse_image(data["observation.images.left_wrist_0_rgb"])
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim == 0 or actions.shape[-1] != UR_ACTION_DIM:
                raise ValueError(f"UR actions must have last dimension {UR_ACTION_DIM}, got {actions.shape}")
            if not np.all(np.isfinite(actions)):
                raise ValueError("UR action values must be finite")
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class AbsoluteTCPActionsToRelative(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        return {**data, "actions": absolute_actions_to_relative(data["state"], data["actions"])}


@dataclasses.dataclass(frozen=True)
class UROutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :UR_ACTION_DIM])}
