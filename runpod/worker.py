#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on 24 GB (RTX 4090/3090/A10G).
LAZY-HOT architecture (workaround: capture boot/model errors and RETURN them).

WHY LAZY: an earlier version loaded models at import -> the worker crashed at
boot, was marked unhealthy, never took jobs, and the boot traceback was
invisible via the API. Now the worker starts fast and healthy; the FIRST job
triggers model init inside the handler, wrapped so any failure is captured and
returned in the job result (readable via /status/{id}).
"""
import json
import os
import threading
import time
import traceback

import runpod

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")

# environment for the ACE-Step app (set default in-case endpoint doesn't pass)
os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
os.environ.setdefault("ACESTEP_API_HOST", "0.0.0.0")
os.environ.setdefault("ACESTEP_API_PORT", "8001")
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")
# Model load crashed with 'Failed to load model with attention' because
# flash-attn-2 is unavailable (needs GPU-local build). Force SDPA/eager.
os.environ.setdefault("ACESTEP_USE_FLASH_ATTENTION", "false")

_init_lock = threading.Lock()
_client = None            # persistent TestClient (kept warm after first init)
_init_error = None        # captured boot error (returned to caller on failure)


def _ensure_warm():
    """Build app + warm models ONCE (lazy on first job). Returns client or raises."""
    global _client, _init_error
    if _client is not None:
        return _client
    with _init_lock:
        if _client is not None:
            return _client
        if _init_error is not None:
            raise RuntimeError(_init_error)
        try:
            print("[worker] initializing ACE-Step app + models (lazy)...", flush=True)
            import torch
            torch.cuda.init()
            from acestep.api_server import create_app
            from fastapi.testclient import TestClient
            app = create_app()
            c = TestClient(app)
            c.__enter__()
            dev = torch.cuda.get_device_name(0)
            _client = c
            print(f"[worker] models warm on {dev}", flush=True)
            return c
        except Exception as e:
            tb = traceback.format_exc()
            print("[worker] INIT FAILED:\n" + tb, flush=True)
            _init_error = f"{type(e).__name__}: {e}\n{tb[-1500:]}"
            raise RuntimeError(_init_error)


def _render(prompt, lyrics, duration_s) -> bytes:
    c = _ensure_warm()
    r = c.post("/release_task", json={
        "prompt": prompt,
        "lyrics": lyrics or "",
        "audio_duration": float(duration_s),
        "guidance_scale": 7.0,
        "shift": 3.0,
    })
    r.raise_for_status()
    data = r.json().get("data", r.json())
    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError("no task_id")
    deadline = time.time() + 1800
    while time.time() < deadline:
        time.sleep(5)
        qr = c.post("/query_result", json={"task_id_list": [task_id]})
        qr.raise_for_status()
        qd = qr.json().get("data", qr.json())
        tasks = qd.get("tasks") if isinstance(qd, dict) else qd
        if isinstance(tasks, list) and tasks:
            task = tasks[0]
        else:
            continue
        status = task.get("status")
        if status == 1:
            break
        if status == 2:
            raise RuntimeError(f"gen failed: {str(task)[:2000]}")
    else:
        raise RuntimeError("gen timeout")
    result = task.get("result", "")
    audio_path = None
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            result = None
    if isinstance(result, list) and result and isinstance(result[0], dict):
        audios = result[0].get("audios")
        if isinstance(audios, list) and audios and isinstance(audios[0], dict):
            audio_path = audios[0].get("path") or audios[0].get("audio_path")
    if not audio_path:
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
        raise RuntimeError("no audio path")
    ar = c.get("/v1/audio", params={"path": audio_path})
    ar.raise_for_status()
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
        tb = traceback.format_exc()
        print("[worker] JOB ERROR:\n" + tb, flush=True)
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}"[:800],
            "workerLogs": tb[-2000:],   # capture+return for /status visibility
        }


runpod.serverless.start({"handler": handler})