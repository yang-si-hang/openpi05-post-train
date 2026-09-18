export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_assets
export OPENPI_CHECKPOINT_ROOT=/root/autodl-tmp/workspace/openpi05-post-train/checkpoints
export WANDB_MODE=online
export WANDB_API_KEY=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

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897

# 为单卡训练准备
unset NCCL_P2P_DISABLE
unset NCCL_LAUNCH_MODE
unset NCCL_DEBUG
unset CUDA_LAUNCH_BLOCKING