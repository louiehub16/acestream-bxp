# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# ACE-Stream 5090 BXP — GPU worker image (SaladCloud RTX 5090 / sm_120 Blackwell)
#
# Build (from project ROOT, so sidecar/ + docker/ are in context):
#   docker build -f docker/Dockerfile -t <user>/acestep-bxp:latest .
#
# CI: .github/workflows/docker-build.yml builds this on GH Actions (amd64 only)
# and pushes to Docker Hub. Salad nodes are x86_64 — do NOT add arm64.
#
# KEY CHOICES (why):
#   * cuda 12.8 base + torch cu128 wheels -> sm_120 (RTX 5090) kernels exist.
#     CUDA 12.1-era images/wheels CANNOT run Blackwell GPUs.
#   * Ubuntu 24.04 ships Python 3.12 = ACE-Step 1.5's supported range (3.11-3.12).
#   * Official ACE-Step API server is the app (no custom inference code).
#   * Weights pull from HF at first boot (HF_HOME=/hf-cache) unless baked below.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

# Pin of ACE-Step/ACE-Step-1.5 main used at build time (overridable).
ARG ACE_STEP_REF=14c0211d5a0653b0f63e27686f4c3f151b4d8629
# Set BAKE_WEIGHTS=true to pre-download model weights into the image (~25 GB
# extra). WARNING: may exceed GitHub runner disk; leave false for Salad where
# nodes pull weights from HF once at boot instead.
ARG BAKE_WEIGHTS=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg openssh-client openssh-server \
        python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# --- uv-managed venv ----------------------------------------------------------
RUN python -m pip install --break-system-packages --no-cache-dir uv

RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git /app/ACE-Step-1.5 \
    && git -C /app/ACE-Step-1.5 checkout "${ACE_STEP_REF}"

WORKDIR /app/ACE-Step-1.5

# 1) Create venv, then LEAN install targeting the API-server runtime only.
#    NOTE: two separate install steps are REQUIRED - torch comes from the cu128
#    index (has ONLY torch), everything else from default PyPI. Using --index-url
#    on the combined line REPLACES PyPI and breaks transformers/fastapi resolution.
RUN uv venv .venv
RUN uv pip install --python .venv/bin/python \
        "torch==2.10.0+cu128" "torchaudio==2.10.0+cu128" --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python .venv/bin/python \
        "transformers>=4.51,<4.58" fastapi "uvicorn[standard]" requests python-multipart \
        boto3 huggingface-hub loguru python-dotenv vector-quantize-pytorch torchao \
        soundfile numpy pydantic safetensors einops starlette tqdm rich pyyaml diskcache xxhash \
        gradio matplotlib pillow diffusers \
        && rm -rf ~/.cache/uv

ENV PATH="/app/ACE-Step-1.5/.venv/bin:${PATH}" \
    HF_HOME=/hf-cache \
    ACESTEP_API_HOST="::" \
    ACESTEP_API_PORT=8001 \
    ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B \
    ACESTEP_LM_BACKEND=pt

# --- OPTIONAL: bake weights at build time --------------------------------------
# Heredoc RUN keeps the python script readable; BuildKit syntax 1.4+ required.
RUN <<'EOF'
set -e
if [ "${BAKE_WEIGHTS}" = "true" ]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download

for repo in (
    "ACE-Step/acestep-v15-xl-turbo",
    "ACE-Step/acestep-v15-xl-sft",
    "ACE-Step/acestep-5Hz-lm-4B",
    "ACE-Step/acestep-5Hz-lm-0.6B",
):
    print(f"[bake] {repo} -> {os.environ.get('HF_HOME', '<default>')}")
    snapshot_download(repo)
print("[bake] done")
PY
else
  echo "[docker] BAKE_WEIGHTS=false -> weights will download from HF on first boot"
fi
EOF

# --- our control files ----------------------------------------------------------
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
COPY sidecar/ /app/sidecar/
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8001

LABEL org.opencontainers.image.title="acestream-bxp" \
      org.opencontainers.image.description="ACE-Step 1.5 bulk music generation worker for SaladCloud RTX 5090" \
      org.opencontainers.image.ace_step_ref="${ACE_STEP_REF}"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
