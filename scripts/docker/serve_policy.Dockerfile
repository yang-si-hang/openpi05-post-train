# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
ARG UV_VERSION=0.11.32
COPY --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
# FFmpeg is required by TorchCodec for decoding LeRobot video datasets.
# git-lfs is retained for Git dependencies that contain LFS pointers.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive \
       apt-get install -y --no-install-recommends \
           ffmpeg \
           git \
           git-lfs \
           linux-headers-generic \
           build-essential \
           clang \
    && rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

# CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
CMD ["/bin/bash", "-lc", "exec uv run --frozen scripts/serve_policy.py $SERVER_ARGS"]

# BEGIN CODEX CLI
USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        git \
        bubblewrap && \
    rm -rf /var/lib/apt/lists/*

# Install the executable package outside /root/.codex.
# /root/.codex will be reserved for persistent runtime state.
RUN curl -fL \
        --retry 5 \
        --retry-delay 2 \
        --retry-all-errors \
        https://raw.githubusercontent.com/openai/codex/main/scripts/install/install.sh \
        -o /tmp/install-codex.sh && \
    CODEX_NON_INTERACTIVE=1 \
    CODEX_INSTALL_DIR=/usr/local/bin \
    CODEX_HOME=/opt/codex \
    CODEX_INSTALLER_USE_RELEASES_OPENAI_COM=false \
        sh /tmp/install-codex.sh && \
    rm -f /tmp/install-codex.sh && \
    /usr/local/bin/codex --version

# END CODEX CLI