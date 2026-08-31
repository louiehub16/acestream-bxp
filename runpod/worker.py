#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on 24 GB (RTX 4090/3090/A10G).
LAZY-HOT + R2 LOG-STREAMING.

- Model init is LAZY (inside first handler) so the worker starts healthy.
- Full stdout/stderr + model-init traceback streamed to R2 (logs/{job}.txt).
- shift=3.0, pt backend, flash-attn off -> SDPA/eager.
"""
import io
import json
import os
import sys
import threading
import time
import traceback

import boto3
import runpod
from fastapi.testclient import TestClient

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")

os.environ.setdefault("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
os.environ.setdefault("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B")
os.environ.setdefault("ACESTEP_LM_BACKEND", "pt")
os.environ.setdefault("ACESTEP_API_HOST", "0.0.0.0")
os.environ.setdefault("ACESTEP_API_PORT", "8001")
os.environ.setdefault("RUNPOD_INIT_TIMEOUT", "800")
os.environ.setdefault("ACESTEP_USE_FLASH_ATTENTION", "false")

_init_lock = threading.Lock()
_client = None
_init_tb = None


def _log_to_r2(key: str, text: str):
    try:
        s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_AKID,
                          aws_secret_access_key=R2_SAK, region_name="auto")
        s3.put_object(Bucket=R2_BUCKET, Key=key, Body=text.encode("utf-8", "replace"),
                      ContentType="text/plain")
    except Exception:
        pass


def _ensure_warm():
    global _client, _init_tb
    if _client is not None:
        return _client
    with _init_lock:
        if _client is not None:
            return _client
        try:
            print("[init] building ACE-Step app + loading models (lazy)...", flush=True)
            import torch
            torch.cuda.init()
            from acestep.api_server import create_app
            app = create_app()
            c = TestClient(app)
            c.__enter__()
            print(f"[init] models warm on {torch.cuda.get_device_name(0)}", flush=True)
            _client = c
            return c
        except Exception:
            _init_tb = traceback.format_exc()
            print("[init] FAILED:\n" + _init_tb, flush=True)
            raise RuntimeError("model init failed (see R2 logs/{job}.txt for full traceback)")


def _render(prompt, lyrics, duration_s) -> bytes:
    c = _ensure_warm()
    r = c.post("/release_task", json={
        "prompt": prompt, "lyrics": lyrics or "", "audio_duration": float(duration_s),
        "guidance_scale": 7.0, "shift": 3.0})
    r.raise_for_status()
    data = r.json().get("data", r.json())
    task_id = data.get("task_id") or data.get("id")
    if not task_id:
        raise RuntimeError("no task_id")
    deadline = time.time() + 1800
    task = None
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
    # result is a list of items with a "file" (audio path) field (mp3/wav)
    if isinstance(result, list) and result:
        for item in result:
            if isinstance(item, dict):
                f = item.get("file") or item.get("path") or item.get("audio_path")
                if f:
                    audio_path = f
                    break
        if not audio_path:
            for item in result:
                if isinstance(item, dict) and isinstance(item.get("audio_paths"), list):
                    for p in item["audio_paths"]:
                        if p:
                            audio_path = p
                            break
    if not audio_path:
        def _find(obj, d=0):
            if d > 6 or obj is None:
                return None
            if isinstance(obj, str) and (obj.lower().endswith((".wav", ".mp3"))):
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
        raise RuntimeError(f"no audio path in result: {str(task)[:400]}")
    ar = c.get("/v1/audio", params={"path": audio_path})
    ar.raise_for_status()
    return ar.content


def _upload(data: bytes, key: str) -> str:
    s3 = boto3.client("s3", endpoint_url=R2_ENDPOINT, aws_access_key_id=R2_AKID,
                      aws_secret_access_key=R2_SAK, region_name="auto")
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType="audio/mpeg")
    return s3.generate_presigned_url("get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=86400)


def handler(job):
    job_id = job.get("id", "unknown")
    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    duration = int(inp.get("duration", 240))
    output = inp.get("output_filename", "song.wav")
    lyrics = inp.get("lyrics", "")
    session = inp.get("session", "")

    old_out, old_err = sys.stdout, sys.stderr
    buf = io.StringIO()
    _log_lock = threading.Lock()

    class _Tee:
        def __init__(self, t_out, t_err, t_buf):
            self._out = t_out
            self._err = t_err
            self._buf = t_buf
        def write(self, s):
            with _log_lock:
                self._buf.write(s)
            try:
                self._out.write(s)
            except Exception:
                pass
            return len(s)
        def flush(self):
            try:
                self._out.flush()
            except Exception:
                pass
    tee = _Tee(old_out, old_err, buf)
    sys.stdout = tee
    sys.stderr = tee

    out = os.path.basename(output.replace("\\", "/"))
    out = "".join(c for c in out if c.isalnum() or c in "._-") or "song.mp3"
    if not out.lower().endswith((".wav", ".mp3")):
        out += ".mp3"
    key = f"{session}/{out}" if session else out

    result = None
    try:
        wav = _render(prompt, lyrics, duration)
        url = _upload(wav, key)
        result = {"status": "success", "r2_key": key, "size": len(wav), "download_url": url}
        print(f"[job {job_id}] SUCCESS uploaded {key} ({len(wav)} bytes)", flush=True)
    except Exception as e:
        tb = traceback.format_exc()
        if _init_tb and "model init failed" in str(e):
            tb = tb + "\n===== MODEL INIT TRACEBACK =====\n" + _init_tb
        print(f"[job {job_id}] ERROR: {tb}", flush=True)
        result = {"status": "error", "error": f"{type(e).__name__}: {e}"[:800],
                  "workerLogs": tb[-4000:]}
    finally:
        sys.stdout, sys.stderr = old_out, old_err
        log_text = buf.getvalue()
        if log_text:
            _log_to_r2(f"logs/{job_id}.txt", log_text)
    return result


runpod.serverless.start({"handler": handler})