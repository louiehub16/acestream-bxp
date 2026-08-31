# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# RunPod Serverless worker — ACE-Step 1.5 xl-turbo on RTX 4090
#
# Build (from project ROOT):
#   docker build -f runpod/Dockerfile -t hrm3478938/acestep-runpod:latest .
#
# CI: push to GitHub, GH Actions builds + pushes (reuse docker-build pattern).
# Image is amd64 only — RunPod serverless workers are x86_64.
#
# Uses the verified LEAN install (the same that shrank the Salad image to
# 10.39 GB): torch cu128 from the torch index, everything else from PyPI,
# skipping gradio/vllm. The worker boots acestep.api_server (pt backend).
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.2-runtime-ubuntu24.04

ARG ACE_STEP_REF=14c0211d5a0653b0f63e27686f4c3f151b4d8629

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg python3 python3-dev python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3

# uv
RUN python3 -m pip install --break-system-packages --no-cache-dir uv

RUN git clone https://github.com/ACE-Step/ACE-Step-1.5.git /workspace \
    && git -C /workspace checkout "${ACE_STEP_REF}"

WORKDIR /workspace

# --- LEAN install (verified): torch cu128 in its own RUN, PyPI deps separate ---
RUN uv venv .venv
RUN uv pip install --python .venv/bin/python \
        torch torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN uv pip install --python .venv/bin/python \
        transformers fastapi "uvicorn[standard]" requests python-multipart \
        boto3 huggingface-hub loguru python-dotenv vector-quantize-pytorch torchao \
        "runpod>=1.9.0" soundfile numpy pydantic safetensors einops starlette tqdm rich pyyaml diskcache xxhash \
        gradio matplotlib pillow \
        && rm -rf ~/.cache/uv

# install the ace-step package itself (no deps - we supply them leanly)
RUN uv pip install --python .venv/bin/python -e . --no-deps

ENV PATH="/workspace/.venv/bin:${PATH}" \
    ACESTEP_WORKDIR=/workspace \
    ACESTEP_API_HOST=0.0.0.0 \
    ACESTEP_API_PORT=8001 \
    ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B \
    ACESTEP_LM_BACKEND=pt \
    ACESTEP_INIT_LLM=true \
    RUNPOD_INIT_TIMEOUT=800

COPY runpod/worker.py /workspace/worker.py
# local-testing convenience: test_input.json drives local `python worker.py`
COPY runpod/test_input.json /workspace/test_input.json

EXPOSE 8001

# RunPod serverless entrypoint: the worker registers the handler and long-polls.
# -u = unbuffered (docs recommend it for serverless logging).
CMD ["python3", "-u", "/workspace/worker.py"]

LABEL org.opencontainers.image.title="acestep-runpod" \
      org.opencontainers.image.description="ACE-Step 1.5 xl-turbo RunPod serverless worker (RTX 4090)" \
      org.opencontainers.image.ace_step_ref="${ACE_STEP_REF}"