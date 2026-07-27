# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.11-slim-bookworm AS mock
COPY --from=uv /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PATH="/app/.venv/bin:$PATH"
COPY services/api/pyproject.toml services/api/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY services/api/neelverse ./neelverse
RUN uv sync --frozen --no-dev
EXPOSE 8000
CMD ["uvicorn", "neelverse.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS gpu
COPY --from=uv /uv /uvx /bin/
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg git libgl1 libglib2.0-0 build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON=3.11 \
    PATH="/app/.venv/bin:$PATH" \
    CUDA_MODULE_LOADING=LAZY \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256" \
    TOKENIZERS_PARALLELISM=false \
    MAX_JOBS=4 \
    FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
RUN uv python install 3.11
COPY services/api/pyproject.toml services/api/uv.lock ./
RUN uv sync --frozen --no-dev --extra gpu --no-install-project
# flash-attn and kernels need CUDA compilation
RUN uv pip install flash-attn --no-build-isolation || echo "WARN: flash-attn build failed, will use sdpa fallback"
RUN uv pip install kernels || echo "WARN: kernels not available, using fallback attention"
RUN uv pip install sageattention || echo "WARN: sageattention not available"
COPY services/api/neelverse ./neelverse
RUN uv sync --frozen --no-dev --extra gpu
EXPOSE 8000
CMD ["uvicorn", "neelverse.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
