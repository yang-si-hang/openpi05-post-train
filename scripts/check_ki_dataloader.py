"""Smoke-test the real KI loader, including spawn workers and batch contracts."""

import dataclasses
import os
import time

import jax
import numpy as np
import psutil
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str = "pi05_ur10e_lora_ki_finetune"
    num_workers: int = 0
    num_batches: int = 3


def main(args: Args) -> None:
    config = dataclasses.replace(_config.get_config(args.config_name), num_workers=args.num_workers)
    started = time.perf_counter()
    loader = _data_loader.create_data_loader(config, num_batches=args.num_batches)
    iterator = iter(loader)
    process = psutil.Process(os.getpid())
    peak_rss = process.memory_info().rss
    batch_times = []
    for index in range(args.num_batches):
        before = time.perf_counter()
        observation, actions = next(iterator)
        jax.block_until_ready(actions)
        batch_times.append(time.perf_counter() - before)
        peak_rss = max(peak_rss, process.memory_info().rss)
        expected = (config.batch_size, config.model.ki_max_token_len)
        fields = {
            "ki_tokens": (observation.ki_tokens, np.int32),
            "ki_token_mask": (observation.ki_token_mask, np.bool_),
            "ki_ar_mask": (observation.ki_ar_mask, np.bool_),
            "ki_loss_mask": (observation.ki_loss_mask, np.bool_),
        }
        for name, (value, dtype) in fields.items():
            assert value is not None, name
            assert value.shape == expected, (name, value.shape)
            assert np.dtype(value.dtype) == np.dtype(dtype), (name, value.dtype)
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
        print(f"batch={index} seconds={batch_times[-1]:.3f}")
    print(
        f"PASS startup_and_batches_seconds={time.perf_counter() - started:.3f} "
        f"batch_seconds={batch_times} main_peak_rss_mib={peak_rss / 2**20:.1f} workers={args.num_workers}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
