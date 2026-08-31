#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on 24 GB (RTX 4090/3090/A10G).
SINGLE-PROCESS, MODELS-WARM architecture (fixed deadlock + cold-start dispatch).

FIXES:
  - Deadlock: no subprocess web-server. We build the ACE-Step FastAPI app
    in-process and keep a persistent TestClient open, so the app's lifespan
    (model download+load + worker daemons) runs ONCE at boot and stays warm.
  - Cold-start dispatch: finding a ready-but-idle worker that never takes a job
    was because the model load was deferred into the lifespan and the worker
    wasn't truly ready. By warming models in GLOBAL scope we guarantee the first
    job hits a ready worker. RUNPOD_INIT_TIMEOUT=800 extends the boot budget.
  - single process / 'pt' LM backend avoids vLLM/vRAM fragmentation on 24GB.

The one persistent TestClient keeps the FastAPI app + lifespan alive for the
whole worker process, so every job rides warm models (no per-request reload).
"""
import json
import os
import time
import traceback

import runpod

# ---------------------------------------------------------------------------
# GLOBAL INIT: build the app + warm the models via its lifespan ONCE.
# ---------------------------------------------------------------------------
print("[worker] warm-boot: building ACE-Step app in-process...", flush=True)

import torch
torch.cuda.init()

os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
os.environ.setdefault("ACESTEP_API_HOST", "0.0.0.0")
os.environ.setdefault("ACESTEP_API_PORT", "8001")
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")   # >7min cold-start budget

from acestep.api_server import create_app
app = create_app()

from fastapi.testclient import TestClient
# Opening the TestClient as a persistent context runs app.lifespan() -> loads
# xl-turbo + LM and starts the API worker daemons, then keeps them alive.
_client = TestClient(app)
_client.__enter__()
print(f"[worker] ACE-Step app warmed on {torch.cuda.get_device_name(0)}", flush=True)

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")


def _render(prompt, lyrics, duration_s) -> bytes:
    """Drive one generation through the warm app (release_task -> poll -> audio)."""
    r = _client.post("/release_task", json={
        "prompt": prompt,
        "lyrics": lyrics or "",
        "audio_duration": float(duration_s),
        "guidance_scale": 7.0,
    })
    if r.status_code != 200:
        raise RuntimeError(f"release_task HTTP {r.status_code}: {r.text[:300]}")
    data = r.json().get("data", r.json())
    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError("no task_id")

    deadline = time.time() + 1800
    result = None
    while time.time() < deadline:
        time.sleep(5)
        qr = _client.post("/query_result", json={"task_id_list": [task_id]})
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

    def find_audio(obj, d=0):
        if d > 8 or obj is None:
            return None
        if isinstance(obj, str) and (obj.lower().endswith((".wav", ".mp3")) or obj.startswith(("/", "http"))):
            return obj
        if isinstance(obj, dict):
            for v in obj.values():
                f = find_audio(v, d + 1)
                if f:
                    return f
        if isinstance(obj, (list, tuple)):
            for v in obj:
                f = find_audio(v, d + 1)
                if f:
                    return f
        return None

    audio_ref = find_audio(result)
    if not audio_ref:
        raise RuntimeError(f"no audio in result: {list(result)[:8]}")
    ar = _client.get("/v1/audio", params={"path": audio_ref})
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