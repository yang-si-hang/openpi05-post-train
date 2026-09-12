# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Convert a JAX Pi0/Pi0.5 LoRA checkpoint to merged OpenPI_RLinf weights.

The effective weights are materialized in the JAX layout before the existing
JAX-to-PyTorch layout transforms run. The output is a normal, non-LoRA
OpenPI_RLinf checkpoint and must be loaded with the base Gemma variants.

Example:
    python -m rlinf.utils.ckpt_convertor.openpi.convert \
        --mode jax_lora_to_openpi_rlinf \
        --input-model data/openpi-checkpoints/pi05_ur10e_lora_train_time_rtc/25000 \
        --input-norm-stats data/openpi-checkpoints/pi05_ur10e_lora_train_time_rtc/25000/assets/pick_v4_merge_crop_vid/norm_stats.json \
        --output-model data/openpi-rlinf-checkpoints/pi05_ur10e_lora_train_time_rtc/25000 \
        --output-norm-stats data/openpi-rlinf-checkpoints/pi05_ur10e_lora_train_time_rtc/25000/assets/pick_v4_merge_crop_vid/norm_stats.json \
        --paligemma-lora-alpha 16 \
        --action-expert-lora-alpha 32
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import torch

from rlinf.utils.ckpt_convertor.openpi import jax_to_openpi_rlinf as base_converter
from rlinf.utils.ckpt_convertor.openpi._core import (
    copy_norm_stats,
    save_safetensors,
    write_config_json,
)

_ATTENTION_MODULES = (
    ("q_einsum", "paligemma"),
    ("kv_einsum", "paligemma"),
    ("attn_vec_einsum", "paligemma"),
    ("q_einsum_1", "action_expert"),
    ("kv_einsum_1", "action_expert"),
    ("attn_vec_einsum_1", "action_expert"),
)
_LORA_KEY_PARTS = ("lora_a", "lora_b", "_lora_a", "_lora_b")


def _unwrap_values(tree: Any) -> Any:
    """Recursively unwrap NNX/Orbax ``{"value": leaf}`` containers."""
    if isinstance(tree, dict):
        if set(tree) == {"value"}:
            return _unwrap_values(tree["value"])
        return {key: _unwrap_values(value) for key, value in tree.items()}
    return tree


def _load_jax_params(checkpoint_dir: str | pathlib.Path) -> dict:
    """Load JAX parameters and normalize NNX variable-state containers."""
    # Current Orbax requires an absolute TensorStore checkpoint path.
    checkpoint_dir = pathlib.Path(checkpoint_dir).expanduser().resolve()
    return _unwrap_values(base_converter._load_jax_params(checkpoint_dir))


def _merge_attention_lora(module: dict, *, alpha: float, path: str) -> np.ndarray:
    """Merge an upstream ``lora.Einsum`` module along its last two axes."""
    if "w" not in module:
        raise ValueError(f"Missing base weight at {path}/w")
    has_a = "lora_a" in module
    has_b = "lora_b" in module
    if has_a != has_b:
        missing = "lora_b" if has_a else "lora_a"
        raise ValueError(f"Incomplete LoRA module {path}: missing {missing}")
    if not has_a:
        raise ValueError(f"Expected LoRA parameters at {path}")

    weight = np.asarray(module["w"], dtype=np.float32)
    lora_a = np.asarray(module["lora_a"], dtype=np.float32)
    lora_b = np.asarray(module["lora_b"], dtype=np.float32)
    if lora_a.shape[:-2] != weight.shape[:-2] or lora_b.shape[:-2] != weight.shape[:-2]:
        raise ValueError(
            f"LoRA batch dimensions do not match at {path}: "
            f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
        )
    if lora_a.shape[-2] != weight.shape[-2] or lora_b.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"LoRA matrix dimensions do not match at {path}: "
            f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
        )
    rank = lora_a.shape[-1]
    if rank <= 0 or lora_b.shape[-2] != rank:
        raise ValueError(
            f"Invalid LoRA rank at {path}: a={lora_a.shape}, b={lora_b.shape}"
        )
    return weight + np.matmul(lora_a, lora_b) * (alpha / rank)


def _merge_attn_vec_lora(module: dict, *, alpha: float, path: str) -> np.ndarray:
    """Merge an attention output projection with OpenPI's einsum semantics.

    For ``BTNH,NHD->BTD``, OpenPI's two LoRA einsums both contract the head
    dimension. Consequently, the equivalent weight update uses every head of
    ``lora_b`` for each head of ``lora_a``, rather than a per-head matmul.
    """
    if "w" not in module:
        raise ValueError(f"Missing base weight at {path}/w")
    has_a = "lora_a" in module
    has_b = "lora_b" in module
    if has_a != has_b:
        missing = "lora_b" if has_a else "lora_a"
        raise ValueError(f"Incomplete LoRA module {path}: missing {missing}")
    if not has_a:
        raise ValueError(f"Expected LoRA parameters at {path}")

    weight = np.asarray(module["w"], dtype=np.float32)
    lora_a = np.asarray(module["lora_a"], dtype=np.float32)
    lora_b = np.asarray(module["lora_b"], dtype=np.float32)
    if weight.ndim < 3 or lora_a.ndim != weight.ndim or lora_b.ndim != weight.ndim:
        raise ValueError(
            f"Invalid attention output LoRA dimensions at {path}: "
            f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
        )
    if lora_a.shape[:-3] != weight.shape[:-3] or lora_b.shape[:-3] != weight.shape[:-3]:
        raise ValueError(
            f"LoRA batch dimensions do not match at {path}: "
            f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
        )
    if (
        lora_a.shape[-3:-1] != weight.shape[-3:-1]
        or lora_b.shape[-3] != weight.shape[-3]
        or lora_b.shape[-1] != weight.shape[-1]
    ):
        raise ValueError(
            f"LoRA matrix dimensions do not match at {path}: "
            f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
        )
    rank = lora_a.shape[-1]
    if rank <= 0 or lora_b.shape[-2] != rank:
        raise ValueError(
            f"Invalid LoRA rank at {path}: a={lora_a.shape}, b={lora_b.shape}"
        )

    lora_b_sum_heads = np.sum(lora_b, axis=-3)
    update = np.einsum("...nhr,...rd->...nhd", lora_a, lora_b_sum_heads, optimize=True)
    if update.shape != weight.shape:
        raise ValueError(
            f"LoRA update shape does not match at {path}: "
            f"w={weight.shape}, update={update.shape}"
        )
    return weight + update * (alpha / rank)


def _merge_ffn_lora(module: dict, *, path: str) -> tuple[np.ndarray, np.ndarray]:
    """Merge upstream OpenPI FFN adapters, whose forward uses unscaled A @ B."""
    pairs = (
        ("gating_einsum", "gating_einsum_lora_a", "gating_einsum_lora_b"),
        ("linear", "linear_lora_a", "linear_lora_b"),
    )
    merged = []
    for weight_key, a_key, b_key in pairs:
        if weight_key not in module:
            raise ValueError(f"Missing base weight at {path}/{weight_key}")
        if (a_key in module) != (b_key in module):
            missing = b_key if a_key in module else a_key
            raise ValueError(f"Incomplete LoRA module {path}: missing {missing}")
        if a_key not in module:
            raise ValueError(f"Expected LoRA parameters at {path}/{weight_key}")

        weight = np.asarray(module[weight_key], dtype=np.float32)
        lora_a = np.asarray(module[a_key], dtype=np.float32)
        lora_b = np.asarray(module[b_key], dtype=np.float32)
        try:
            update = np.matmul(lora_a, lora_b)
        except ValueError as exc:
            raise ValueError(
                f"LoRA matrix dimensions do not match at {path}/{weight_key}: "
                f"w={weight.shape}, a={lora_a.shape}, b={lora_b.shape}"
            ) from exc
        if update.shape != weight.shape:
            raise ValueError(
                f"LoRA update shape does not match at {path}/{weight_key}: "
                f"w={weight.shape}, update={update.shape}"
            )
        merged.append(weight + update)
    return merged[0], merged[1]


def _find_lora_paths(tree: Any, prefix: tuple[str, ...] = ()) -> set[str]:
    """Return LoRA leaves so unsupported adapters cannot be dropped silently."""
    if not isinstance(tree, dict):
        return set()
    paths: set[str] = set()
    for key, value in tree.items():
        path = (*prefix, key)
        if any(part in key for part in _LORA_KEY_PARTS):
            paths.add("/".join(path))
        paths.update(_find_lora_paths(value, path))
    return paths


def _merged_llm_params(
    params: dict,
    *,
    paligemma_lora_alpha: float,
    action_expert_lora_alpha: float,
) -> dict:
    """Replace LoRA modules by effective weights in place.

    Mutating is intentional: a Pi0.5 parameter tree is several GiB, so copying
    the complete tree can exceed the memory available during conversion.
    """
    layers = params["PaliGemma"]["llm"]["layers"]
    attention = layers["attn"]
    for name, expert in _ATTENTION_MODULES:
        alpha = (
            paligemma_lora_alpha if expert == "paligemma" else action_expert_lora_alpha
        )
        path = f"PaliGemma/llm/layers/attn/{name}"
        merge = (
            _merge_attn_vec_lora
            if name.startswith("attn_vec_einsum")
            else _merge_attention_lora
        )
        attention[name] = {"w": merge(attention[name], alpha=alpha, path=path)}

    for name in ("mlp", "mlp_1"):
        gating, linear = _merge_ffn_lora(
            layers[name], path=f"PaliGemma/llm/layers/{name}"
        )
        layers[name] = {"gating_einsum": gating, "linear": linear}

    remaining = _find_lora_paths(params)
    if remaining:
        raise ValueError(
            f"Unsupported or unmerged LoRA parameters: {sorted(remaining)}"
        )
    return params


def convert_llm(
    params: dict,
    pi05: bool,
    *,
    paligemma_lora_alpha: float = 16.0,
    action_expert_lora_alpha: float = 32.0,
) -> dict[str, torch.Tensor]:
    """Merge all dual-expert LoRA weights and convert the resulting LLM."""
    merged = _merged_llm_params(
        params,
        paligemma_lora_alpha=paligemma_lora_alpha,
        action_expert_lora_alpha=action_expert_lora_alpha,
    )
    return base_converter.convert_llm(merged, pi05)


def convert(
    input_model: str | pathlib.Path,
    input_norm_stats: str | pathlib.Path,
    output_model: str | pathlib.Path,
    output_norm_stats: str | pathlib.Path,
    *,
    pi05: bool = True,
    action_dim: int = 32,
    action_horizon: int = 20,
    max_token_len: int = 200,
    paligemma_lora_alpha: float = 16.0,
    action_expert_lora_alpha: float = 32.0,
    pcd: bool = False,
    dtype: str = "bfloat16",
) -> pathlib.Path:
    """Convert JAX LoRA weights into a merged, non-LoRA OpenPI_RLinf model."""
    output_model = pathlib.Path(output_model)
    params = _load_jax_params(input_model)

    converted: dict[str, torch.Tensor] = {}
    for part in (
        base_converter.convert_siglip(params),
        convert_llm(
            params,
            pi05,
            paligemma_lora_alpha=paligemma_lora_alpha,
            action_expert_lora_alpha=action_expert_lora_alpha,
        ),
        base_converter.convert_projections(params, pi05),
    ):
        converted.update({key: value.contiguous() for key, value in part.items()})

    save_safetensors(converted, output_model / "model.safetensors")
    write_config_json(
        {
            "action_dim": action_dim,
            "action_horizon": action_horizon,
            "max_token_len": max_token_len,
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
            "pi05": pi05,
            "pcd": pcd,
            "dtype": dtype,
        },
        output_model,
    )
    copy_norm_stats(input_norm_stats, output_norm_stats)
    return output_model


def add_arguments(parser) -> None:
    """Register the ``jax_lora_to_openpi_rlinf`` mode arguments."""
    parser.add_argument("--input-model", required=True, help="JAX checkpoint directory")
    parser.add_argument("--input-norm-stats", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-norm-stats", required=True)
    parser.add_argument("--no-pi05", dest="pi05", action="store_false")
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--action-horizon", type=int, default=20)
    parser.add_argument("--max-token-len", type=int, default=200)
    parser.add_argument("--paligemma-lora-alpha", type=float, default=16.0)
    parser.add_argument("--action-expert-lora-alpha", type=float, default=32.0)


def run(args) -> None:
    """Execute the ``jax_lora_to_openpi_rlinf`` mode."""
    convert(
        args.input_model,
        args.input_norm_stats,
        args.output_model,
        args.output_norm_stats,
        pi05=args.pi05,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        max_token_len=args.max_token_len,
        paligemma_lora_alpha=args.paligemma_lora_alpha,
        action_expert_lora_alpha=args.action_expert_lora_alpha,
    )
