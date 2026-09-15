# --- UniRig RunPod Serverless worker image ---
# Base: RunPod's official CUDA/PyTorch image. VERIFY this tag still exists at
# https://hub.docker.com/r/runpod/pytorch/tags before building — RunPod
# periodically retires old tags. Pick one that ships CUDA 12.1+ and Python 3.11.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# --- system deps needed by trimesh/pyrender/blender addon etc. ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxi6 libxrandr2 libxfixes3 libxcursor1 libxinerama1 libxkbcommon0 && \
    rm -rf /var/lib/apt/lists/*

# --- clone UniRig ---
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git /workspace/UniRig
WORKDIR /workspace/UniRig

# --- python deps, following the official UniRig install steps ---
# NOTE: torch/torchvision уже есть в базовом образе runpod/pytorch:2.4.0-...
# Переустанавливать их без версии НЕЛЬЗЯ — pip подтянет последний torch с PyPI
# (2.5/2.6/...), затрёт torch 2.4.0, и torch_scatter/torch_cluster (собранные
# ниже строго под torch-2.4.0+cu124) перестанут грузиться:
#   OSError: ... undefined symbol: _ZN5torch3jit17parseSchemaOrNameERKSsb
# Если requirements.txt реально требует другую версию torch — пиши её явно и
# синхронно с версией в -f https://data.pyg.org/whl/torch-X.Y.Z+cuXXX.html ниже.
RUN python -m pip install --upgrade pip

# requirements.txt lists flash-attn, which fails to build here because torch
# isn't fully set up yet at this point. Strip it out and install it as its
# own step further down instead.
RUN sed -i '/flash[_-]attn/Id' requirements.txt

RUN python -m pip install --ignore-installed blinker -r requirements.txt && \
    python -m pip install numpy==1.26.4

# spconv — pick the wheel matching your CUDA major version (cu124 here to match the base image)
RUN python -m pip install spconv-cu124

# torch_scatter / torch_cluster — must match the torch+cuda combo above
RUN python -m pip install torch_scatter torch_cluster \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# flash-attn is notoriously fragile to build. Try the prebuilt wheel; if it's
# not available for this torch/cuda/python combo, don't fail the whole build —
# fall back and check at runtime whether UniRig still works without it.
RUN python -m pip install flash-attn --no-build-isolation || \
    echo "flash-attn wheel not available for this combo — continuing without it"

# --- serverless glue ---
RUN python -m pip install runpod huggingface_hub requests

# --- local test server deps (не обязательны для прод-воркера, но легковесны) ---
RUN python -m pip install fastapi "uvicorn[standard]" python-multipart

# --- prefetch model weights at BUILD time so cold starts don't re-download them ---
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('VAST-AI/UniRig', local_dir='/workspace/UniRig/.cache/unirig_ckpt')"

COPY handler.py /workspace/UniRig/handler.py
COPY local_test_api.py /workspace/UniRig/local_test_api.py

# По умолчанию контейнер стартует как RunPod serverless worker.
# Для локального теста своими моделями (без RunPod) запусти контейнер так:
#   docker run --gpus all -p 8002:8002 --entrypoint python <image> local_test_api.py
# и открой http://localhost:8002/docs
EXPOSE 8002

CMD ["python", "-u", "handler.py"]
