# =============================================================================
# UniRig RunPod Serverless Worker
# ("One Model to Rig Them All" — VAST-AI-Research/UniRig, SIGGRAPH'25)
#
# ПОЧЕМУ ИМЕННО ТАК (важно при отладке, если что-то отвалится на билде):
#
# 1) Python 3.11 — это НЕ прихоть, а жёсткое требование. UniRig тянет
#    `bpy==4.2` (headless Blender как python-пакет) из requirements.txt,
#    а официальные wheel'ы `bpy` на PyPI собраны ТОЛЬКО под cp310/cp311
#    linux_x86_64. На 3.12/3.10 без danger'ов не поставится вовсе.
#
# 2) CUDA "devel" (а не "runtime") образ — нужен nvcc/ninja, т.к.
#    torch_scatter / torch_cluster / (опционально) flash-attn могут
#    доехать до локальной сборки, если подходящего прекомпилированного
#    wheel'а не найдётся под конкретную связку torch+cuda+python.
#
# 3) Порядок установки КРИТИЧЕН и повторяет official README один в один:
#        torch/torchvision -> requirements.txt -> spconv -> torch_scatter/
#        torch_cluster (wheel'ы с data.pyg.org, привязаны к torch+cuda) ->
#        numpy==1.26.4 В САМОМ КОНЦЕ, ПОСЛЕ ВООБЩЕ ВСЕХ pip install
#        (иначе более новый numpy, притянутый другими пакетами — включая
#        безобидные на вид runpod/requests/huggingface_hub — молча
#        переустанавливает numpy и ломает spconv/scipy/torch ABI в рантайме,
#        без единой ошибки на этапе сборки).
#
# 4) flash_attn — самый капризный пакет во всей цепочке. Сначала пробуем
#    официальный precompiled wheel Dao-AILab (быстро, ~10 сек), и только
#    если под конкретную связку его нет / изменился ABI — падаем в сборку
#    из исходников (`--no-build-isolation`, MAX_JOBS ограничен, чтобы не
#    улететь по RAM на раннере GitHub Actions).
#
# 5) Веса модели (skeleton + skin) качаются ЗАРАНЕЕ на этапе билда через
#    huggingface_hub, чтобы cold start воркера на RunPod не тратил время
#    (и не падал по таймауту) на первый инференс. Репозиторий VAST-AI/UniRig
#    на HF ПУБЛИЧНЫЙ (MIT), поэтому HF_TOKEN не обязателен — но параметр
#    оставлен на случай приватного форка/рейт-лимитов HF.
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0" \
    MAX_JOBS=4 \
    PYOPENGL_PLATFORM=egl \
    PYTHONDONTWRITEBYTECODE=1

# --- Системные зависимости -------------------------------------------------
# software-properties-common -> add-apt-repository (deadsnakes для py3.11)
# libx11-6/libxi6/libxrender1/... -> нужны headless Blender'у (bpy) и pyrender
# libxkbcommon0 / libxkbcommon-x11-0 -> без них `import bpy` падает с
#    "libxkbcommon.so.0: cannot open shared object file" (проверено на логах)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl wget git ninja-build build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-distutils \
        libgl1 libglu1-mesa libglib2.0-0 libsm6 libxext6 libxrender1 \
        libxi6 libxxf86vm1 libxfixes3 libxrandr2 libxinerama1 \
        libosmesa6 libegl1 libopengl0 \
        libxkbcommon0 libxkbcommon-x11-0 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /workspace

# --- PyTorch (cu121, тот же мажор что и база образа) ------------------------
RUN pip install --upgrade pip && \
    pip install -U setuptools==69.5.1 wheel packaging ninja && \
    pip install torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cu121

# --- Клонируем официальный репозиторий UniRig -------------------------------
RUN git clone --depth 1 https://github.com/VAST-AI-Research/UniRig.git /workspace/UniRig

WORKDIR /workspace/UniRig

# --- Основные python-зависимости из requirements.txt репозитория -----------
# (transformers, bpy, flash_attn, trimesh, open3d, pyrender, etc.)
# flash_attn и bpy временно вырезаем из общего списка — ставим их отдельно
# ниже с собственной, более надёжной логикой (см. пункты 4 и комментарий
# про bpy). Так один сорвавшийся пакет не рушит установку всех остальных.
# --ignore-installed нужен из-за пакетов вроде `blinker`, которые в базовом
# Ubuntu-образе стоят как distutils-installed (через apt) — pip не умеет их
# аккуратно апгрейдить/удалять ("Cannot uninstall blinker 1.4 ... distutils
# installed project") и падает с exit code 1 без этого флага.
RUN grep -v -E '^(flash_attn|bpy==)' requirements.txt > requirements_main.txt \
    && pip install --ignore-installed -r requirements_main.txt

# --- bpy (headless Blender) --------------------------------------------------
RUN pip install bpy==4.2

# --- spconv: КОНКРЕТНО под cu121, версия должна совпасть с cuda-мажором ----
RUN pip install spconv-cu121

# --- torch_scatter / torch_cluster: wheel'ы жёстко привязаны к торч+cuda ---
RUN pip install torch_scatter torch_cluster \
        -f https://data.pyg.org/whl/torch-2.3.1+cu121.html --no-cache-dir

# --- flash_attn: сначала пытаемся готовый wheel, иначе собираем из исходников
RUN pip install \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    || pip install flash-attn==2.5.8 --no-build-isolation

# --- Наши доп. зависимости под RunPod ---------------------------------------
RUN pip install runpod requests huggingface_hub

# --- (опционально) VRM addon для Blender: нужен только если на вход/выход --
# --- когда-нибудь придут/уйдут .vrm файлы. Не критично, но недорого поставить.
RUN python3 -c "\
import bpy, os; \
bpy.ops.preferences.addon_install(filepath=os.path.abspath('blender/add-on-vrm-v2.20.77_modified.zip'))" \
    || echo "[UniRig] VRM addon install skipped (non-fatal)"

# --- Известный гочтя из README: pyrender иногда не находит дисплей внутри --
# --- headless контейнера -> переключаем voxel_skin backend на open3d.      --
# --- sed молча ничего не делает, если строка не найдена — билд не падает.  --
RUN find configs -type f -name '*.yaml' -print0 | \
    xargs -0 sed -i 's/backend:\s*pyrender/backend: open3d/g' || true

# --- Качаем веса модели ЗАРАНЕЕ (на этапе билда), чтобы cold start был быстрым
# HF_TOKEN не обязателен (репозиторий публичный), но пробрасываем на случай
# приватного зеркала/рейт-лимитов Hugging Face.
ARG HF_TOKEN=""
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
tok = '${HF_TOKEN}' or None; \
hf_hub_download(repo_id='VAST-AI/UniRig', filename='skeleton/articulation-xl_quantization_256/model.ckpt', token=tok); \
hf_hub_download(repo_id='VAST-AI/UniRig', filename='skin/articulation-xl/model.ckpt', token=tok); \
print('[UniRig] checkpoints cached at build time.')"

# --- numpy СТРОГО ПОСЛЕДНИМ pip install в образе -----------------------------
# Важно: этот шаг должен идти ПОСЛЕ вообще всех остальных pip install выше
# (в т.ч. после runpod/requests/huggingface_hub), иначе они молча притянут
# другой numpy как транзитивную зависимость и сломают ABI scipy/spconv/torch
# в рантайме (именно это было причиной "numpy._core.multiarray failed to
# import" / "AttributeError: numpy._globals ... _signature_descriptor").
# --no-deps не даёт pip заново резолвить зависимости numpy.
RUN pip install --ignore-installed --no-deps numpy==1.26.4

# --- Копируем наш handler ----------------------------------------------------
COPY handler.py /workspace/UniRig/handler.py

WORKDIR /workspace/UniRig

CMD ["python3", "-u", "handler.py"]
