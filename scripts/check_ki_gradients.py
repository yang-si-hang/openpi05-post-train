"""Validate KI branch gradient insulation on one real transformed batch."""

import dataclasses
import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_ur10e_lora_ki_finetune"


def _group(path: tuple[object, ...]) -> str:
    name = "/".join(map(str, path))
    if "/img/" in f"/{name}/":
        return "image_encoder"
    if "llm" in name and "_1" not in name:
        return "vlm_lora"
    if "llm" in name and "_1" in name:
        return "action_expert_lora"
    if "action_in_proj" in name:
        return "action_in_proj"
    if "time_mlp" in name:
        return "time_mlp"
    if "action_out_proj" in name:
        return "action_out_proj"
    return "other_trainable"


def _report(label: str, grads: nnx.State) -> dict[str, float]:
    grouped: dict[str, list[object]] = {}
    for path, variable in grads.flat_state().items():
        grouped.setdefault(_group(path), []).append(variable.value)
        if _group(path) == "other_trainable":
            print(f"UNCLASSIFIED {path}")
    if grouped.get("other_trainable"):
        raise RuntimeError("Unclassified trainable parameter paths")
    norms = {name: float(optax.global_norm(values)) for name, values in grouped.items()}
    if not all(math.isfinite(value) for value in norms.values()):
        raise RuntimeError(f"Non-finite {label} gradients: {norms}")
    print(label, norms)
    return norms


def main(args: Args) -> None:
    config = dataclasses.replace(_config.get_config(args.config_name), batch_size=1, num_workers=0)
    loader = _data_loader.create_data_loader(config, num_batches=1)
    observation, actions = next(iter(loader))
    # The base checkpoint does not contain the LoRA parameters introduced by
    # this config. Start from a real model so the weight loader can preserve
    # concrete initialized arrays for every missing LoRA leaf. Using an
    # eval_shape model here would leave ShapeDtypeStruct values in the model,
    # which cannot participate in value_and_grad.
    model = config.model.create(jax.random.key(0))
    params = config.weight_loader.load(nnx.state(model).to_pure_dict())
    if any(isinstance(value, jax.ShapeDtypeStruct) for value in jax.tree.leaves(params)):
        raise RuntimeError("Weight loading left abstract ShapeDtypeStruct parameters in the model")
    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(params)
    model = nnx.merge(graphdef, state)
    model.train()
    processed = _model.preprocess_observation(jax.random.key(1), observation, train=True)

    def fast_loss(module: _pi0.Pi0):
        encoded_images = module._encode_images(processed)  # noqa: SLF001
        prefix_out, prefix_mask, kv_cache = module._forward_prefix(  # noqa: SLF001
            processed, encoded_images=encoded_images
        )
        return jnp.mean(
            module._compute_fast_loss_from_shared_prefix(  # noqa: SLF001
                processed,
                prefix_out=prefix_out,
                prefix_mask=prefix_mask,
                kv_cache=kv_cache,
            )[0]
        )

    def flow_loss(module: _pi0.Pi0):
        encoded_images = module._encode_images(processed)  # noqa: SLF001
        _, prefix_mask, kv_cache = module._forward_prefix(processed, encoded_images=encoded_images)  # noqa: SLF001
        flow_kv = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        return jnp.mean(
            module._compute_flow_loss_from_prefix_cache(  # noqa: SLF001
                jax.random.key(2),
                jax.random.key(3),
                None,
                processed,
                actions,
                prefix_mask=prefix_mask,
                kv_cache=flow_kv,
            )
        )

    diff = nnx.DiffState(0, config.trainable_filter)
    _, fast_grads = nnx.value_and_grad(fast_loss, argnums=diff)(model)
    _, flow_grads = nnx.value_and_grad(flow_loss, argnums=diff)(model)
    fast = _report("FAST-only", fast_grads)
    flow = _report("FM-only-insulated", flow_grads)
    fast_expected = ("image_encoder", "vlm_lora")
    flow_expected = ("action_expert_lora", "action_in_proj", "time_mlp", "action_out_proj")
    assert all(fast.get(name, 0.0) > 0 for name in fast_expected)
    assert all(fast.get(name, 0.0) == 0 for name in flow_expected)
    assert all(flow.get(name, 0.0) == 0 for name in fast_expected)
    assert all(flow.get(name, 0.0) > 0 for name in flow_expected)
    print("PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
