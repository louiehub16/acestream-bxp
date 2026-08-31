#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on RTX 4090.
SINGLE-PROCESS / NO-SUBPROCESS architecture.

FIXES the previous deadlock: booting `acestep.api_server` as a subprocess
inside a serverless worker blocked RunPod's stdout/queue handshake and the job
stuck IN_QUEUE forever. Here we instead:

  1. Build the ACE-Step FastAPI app IN-PROCESS via `create_app()` at container
     boot (global scope). This triggers the heavy model download+load ONCE,
     using the repo's own tested initialization.
  2. Run one generation through the app's own ASGI callable (fully tested
     pipeline: parse -> store -> run_blocking_generate -> result) - no
     subprocess, no deadlock.
  3. Upload the resulting WAV to Cloudflare R2 + return a presigned link.

GPU: RTX 4090 (24 GB). xl-turbo DiT + 5Hz LM fit in 24GB with LLM backend 'pt'
(PyTorch native, single process) - avoids vLLM's separate memory mapping which
fragments on consumer cards.

IMPORTANT: shift=3.0 for xl-turbo timestep scaling (default 1.0 -> silence/hiss).
Cold start: model download+load ~30-60s -> set RunPod container boot timeout >=120s.
"""
import json
import os
import traceback

import runpod

# ---------------------------------------------------------------------------
# FITNESS CHECK (docs: serverless/development/fitness-checks): Runpod runs
# registered checks BEFORE the worker takes traffic. Validates the GPU is
# usable so a broken worker is marked unhealthy + replaced, not failing jobs.
# Requires runpod>=1.9.0. Runs once at startup then is skipped.
# ---------------------------------------------------------------------------
@runpod.serverless.register_fitness_check
def check_gpu_available():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gb < 8:
        raise RuntimeError(f"GPU VRAM insufficient: {gb:.1f}GB")


# ---------------------------------------------------------------------------
# GLOBAL INIT: build the ACE-Step app once at container boot (not per request).
# This downloads/loads xl-turbo + LM once and keeps them resident.
# ---------------------------------------------------------------------------
print("[worker] building ACE-Step app in-process (global init)...", flush=True)

import torch
torch.cuda.init()

# env: force pt backend, xl-turbo config, 0.6B LM (fits 24GB)
os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
os.environ.setdefault("ACESTEP_API_HOST", "0.0.0.0")
os.environ.setdefault("ACESTEP_API_PORT", "8001")

# KEY (docs: development/optimization): extend the worker's cold-start budget so
# the model download+load (~30-60s+) isn't misread as dead. Without this a worker
# that takes >7 min to boot is marked unhealthy and never takes jobs.
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")

from acestep.api_server import create_app
app = create_app()          # builds handlers + wires run_blocking_generate
print(f"[worker] app built on {torch.cuda.get_device_name(0)}", flush=True)

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")


def _render(prompt, lyrics, duration_s, shift=3.0, guidance=7.0) -> bytes:
    """Run one generation in-process via the app's own generation path.
    Returns raw WAV bytes. Uses the app's method that maps a request to audio."""
    import asyncio
    from fastapi.testclient import TestClient
    from acestep.api.http.release_task_models import GenerateMusicRequest

    # Build the request object the app's route expects
    req = GenerateMusicRequest(
        prompt=prompt,
        lyrics=lyrics or "",
        audio_duration=float(duration_s),
        guidance_scale=guidance,
    )
    # The app exposes the blocking generation; use TestClient to drive /release_task
    # synchronously (in-process, no subprocess, no real network).
    with TestClient(app) as client:
        r = client.post("/release_task", json={
            "prompt": prompt,
            "lyrics": lyrics or "",
            "audio_duration": float(duration_s),
            "guidance_scale": guidance,
        })
        if r.status_code != 200:
            raise RuntimeError(f"release_task HTTP {r.status_code}: {r.text[:300]}")
        data = r.json().get("data", r.json())
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise RuntimeError("no task_id")

        # poll query_result (in-process)
        import time
        deadline = time.time() + 1500
        result = None
        while time.time() < deadline:
            time.sleep(5)
            qr = client.post("/query_result", json={"task_id_list": [task_id]})
            qd = qr.json().get("data", qr.json())
            tasks = qd.get("tasks") if isinstance(qd, dict) else None
            task = (tasks[0] if tasks else None) or (
                qd[0] if isinstance(qd, list) and qd else None)
            if task is None:
                continue
            st = task.get("status")
            if st == 1:
                result = task
                break
            if st == 2:
                raise RuntimeError(f"gen failed: {task}")
        if result is None:
            raise RuntimeError("gen timeout")
        # find audio path
        def find_audio(obj, d=0):
            if d > 8 or obj is None:
                return None
            if isinstance(obj, str) and (obj.lower().endswith((".wav",".mp3")) or obj.startswith(("/","http"))):
                return obj
            if isinstance(obj, dict):
                for v in obj.values():
                    f = find_audio(v, d+1)
                    if f: return f
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    f = find_audio(v, d+1)
                    if f: return f
            return None
        audio_ref = find_audio(result)
        if not audio_ref:
            raise RuntimeError(f"no audio in result: {list(result)[:8]}")
        # download via app's audio route (in-process)
        ar = client.get(f"/v1/audio", params={"path": audio_ref})
        if ar.status_code != 200:
            raise RuntimeError(f"/v1/audio HTTP {ar.status_code}")
        return ar.content


def _upload(data: bytes, key: str) -> str:
    import boto3
    s3 = boto3.client(
        "s3", endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_AKID, aws_secret_access_key=R2_SAK, region_name="auto")
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType="audio/wav")
    return s3.generate_presigned_url("get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=86400)


def handler(job):
    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    duration = int(inp.get("duration", 240))
    output = inp.get("output_filename", "song.wav")
    lyrics = inp.get("lyrics", "")
    session = inp.get("session", "")

    out = os.path.basename(output.replace("\\", "/"))
    out = "".join(c for c in out if c.isalnum() or c in "._-") or "song.wav"
    if not out.lower().endswith(".wav"):
        out += ".wav"
    key = f"{session}/{out}" if session else out

    try:
        wav = _render(prompt, lyrics, duration)
        url = _upload(wav, key)
        return {"status": "success", "r2_key": key, "size": len(wav), "download_url": url}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:800]}


runpod.serverless.start({"handler": handler})