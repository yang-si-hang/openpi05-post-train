"""Scan KI training batches for token lengths, masks, dtypes, and finite model inputs."""

import dataclasses
import math

import jax
import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_ur10e_lora_ki_finetune"
    num_samples: int | None = 5000
    seed: int = 0
    num_examples: int = 3


def main(args: Args) -> None:
    count = args.num_samples or 19_908
    batch_size = min(64, count)
    config = dataclasses.replace(
        _config.get_config(args.config_name), batch_size=batch_size, num_workers=12, seed=args.seed
    )
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=math.ceil(count / batch_size))
    lengths, targets = [], []
    zero_masks = invalid_masks = nonfinite = 0
    seen = 0
    for _batch_index, (obs, actions) in enumerate(loader):
        jax.block_until_ready(actions)
        remaining = min(actions.shape[0], count - seen)
        token_masks = np.asarray(obs.ki_token_mask[:remaining])
        loss_masks = np.asarray(obs.ki_loss_mask[:remaining])
        lengths.extend(token_masks.sum(axis=1).astype(int).tolist())
        targets.extend(loss_masks[:, 1:].sum(axis=1).astype(int).tolist())
        zero_masks += int(np.sum(~loss_masks[:, 1:].any(axis=1)))
        invalid_masks += int(np.sum(loss_masks & ~token_masks))
        nonfinite += int(
            not (
                np.isfinite(np.asarray(obs.state[:remaining])).all()
                and np.isfinite(np.asarray(actions[:remaining])).all()
            )
        )
        for offset in range(min(remaining, max(args.num_examples - seen, 0))):
            print(
                f"example={seen + offset} length={lengths[seen + offset]} targets={targets[seen + offset]} "
                f"boundary={np.flatnonzero(loss_masks[offset])[:3]}"
            )
        seen += remaining
        if seen >= count:
            break
    values = np.asarray(lengths)
    target_values = np.asarray(targets)
    print(
        "samples={} token_length_mean/p90/p95/p99/max={:.2f}/{:.2f}/{:.2f}/{:.2f}/{} "
        "postfix_targets_mean/min/max={:.2f}/{}/{} zero_masks={} invalid_masks={} nonfinite={}".format(
            count,
            values.mean(),
            *np.percentile(values, [90, 95, 99]),
            values.max(),
            target_values.mean(),
            target_values.min(),
            target_values.max(),
            zero_masks,
            invalid_masks,
            nonfinite,
        )
    )
    if zero_masks or invalid_masks or nonfinite:
        raise SystemExit("FAIL")
    print("PASS")


if __name__ == "__main__":
    main(tyro.cli(Args))
