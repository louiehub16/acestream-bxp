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

# 1) locked dependency set from the repo's uv.lock
RUN uv sync --frozen --no-dev

# 2) CUDA-12.8 torch builds (sm_120 need). --reinstall-package guarantees the
#    cu128 wheel REPLACES the lock-pinned build installed by `uv sync`: plain
#    `uv pip install` sees torch/torchaudio already satisfied and would
#    silently no-op, leaving non-sm_120 kernels baked into the image.
RUN uv pip install --python .venv/bin/python --reinstall-package torch --reinstall-package torchaudio \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 3) runtime deps of our sidecar + weight baking
RUN uv pip install --python .venv/bin/python boto3 requests huggingface_hub

ENV PATH="/app/ACE-Step-1.5/.venv/bin:${PATH}" \
    HF_HOME=/hf-cache \
    ACESTEP_API_HOST="::" \
    ACESTEP_API_PORT=8001 \
    ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B \
    ACESTEP_LM_BACKEND=vllm

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
