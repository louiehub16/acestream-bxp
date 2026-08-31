#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on 24 GB (RTX 4090/3090/A10G).
SINGLE-PROCESS, MODELS-WARM architecture (fixed deadlock + cold-start dispatch).

gptsol review fixes incorporated (see review at %TEMP%/gptsol_worker_review.txt):
 - shift=3.0 sent explicitly (ACE-Step default is already 3.0, make it explicit).
 - /query_result parsed with INT status (0=running,1=success,2=failed) per the
   true API (query_result_service.py), not guessed.
 - audio extracted from result.audios[].path (the documented output field), not
   a recursive find_audio.
 - process-wide lock around _render (RunPod can invoke concurrent jobs).
 - R2 env validated at boot; fail fast.
"""
import json
import os
import threading
import time
import traceback

import boto3
import runpod
from fastapi.testclient import TestClient

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
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")

# Validate R2 env at boot (gptsol #11): fail fast, don't waste a full render.
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")
for _k, _v in [("R2_ENDPOINT_URL", R2_ENDPOINT), ("R2_ACCESS_KEY_ID", R2_AKID),
               ("R2_SECRET_ACCESS_KEY", R2_SAK)]:
    if not _v:
        raise RuntimeError(f"missing required env {_k}")


from acestep.api_server import create_app
app = create_app()

# Persistent TestClient runs app.lifespan() once -> loads xl-turbo + LM, starts
# the API worker daemons, and keeps them alive across the whole worker process.
_client = TestClient(app)
_client.__enter__()
print(f"[worker] ACE-Step app warmed on {torch.cuda.get_device_name(0)}", flush=True)

_render_lock = threading.Lock()


def _render(prompt, lyrics, duration_s) -> bytes:
    """Drive one generation through the warm app. Serialized via _render_lock."""
    with _render_lock:
        r = _client.post("/release_task", json={
            "prompt": prompt,
            "lyrics": lyrics or "",
            "audio_duration": float(duration_s),
            "guidance_scale": 7.0,
            "shift": 3.0,          # turbo timestep factor (default is 3.0; make explicit)
        })
        r.raise_for_status()
        data = r.json().get("data", r.json())
        task_id = data.get("task_id") or data.get("id")
        if not task_id:
            raise RuntimeError("no task_id")

        deadline = time.time() + 1800
        while time.time() < deadline:
            time.sleep(5)
            qr = _client.post("/query_result", json={"task_id_list": [task_id]})
            qr.raise_for_status()
            qd = qr.json().get("data", qr.json())
            tasks = qd.get("tasks") if isinstance(qd, dict) else qd
            if isinstance(tasks, list) and tasks:
                task = tasks[0]
            else:
                continue
            status = task.get("status")
            # int status: 0=running, 1=success, 2=failed (per query_result_service.py)
            if status == 1:
                break
            if status == 2:
                raise RuntimeError(f"gen failed: {str(task)[:300]}")
        else:
            raise RuntimeError("gen timeout")

        # extract the documented output audio path (result.audios[].path)
        result = task.get("result", "")
        audio_path = None
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except Exception:
                result = None
        if isinstance(result, list) and result:
            audios = result[0].get("audios") if isinstance(result[0], dict) else None
            if isinstance(audios, list) and audios and isinstance(audios[0], dict):
                audio_path = audios[0].get("path") or audios[0].get("audio_path")
        if not audio_path:
            # fallback: scan result for any .wav path
            def _find(obj, d=0):
                if d > 6 or obj is None:
                    return None
                if isinstance(obj, str) and obj.lower().endswith(".wav"):
                    return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        f = _find(v, d + 1)
                        if f:
                            return f
                if isinstance(obj, list):
                    for v in obj:
                        f = _find(v, d + 1)
                        if f:
                            return f
                return None
            audio_path = _find(result)
        if not audio_path:
            raise RuntimeError("no audio path in result")

        ar = _client.get("/v1/audio", params={"path": audio_path})
        ar.raise_for_status()
        return ar.content


def _upload(data: bytes, key: str) -> str:
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