#!/usr/bin/env python3
"""
RunPod Serverless Worker — ACE-Step 1.5 xl-turbo on RTX 4090.

Each serverless request = ONE song. Uses the VERIFIED path the Salad sidecar
uses: boot the ACE-Step API server (`acestep.api_server`) as a subprocess on
localhost, POST /release_task, poll /query_result, download /v1/audio, upload
to R2. No hand-rolled pipeline calls (the repo's direct API is deep internal
plumbing - dit_handler/llm_handler/GenerationParams).

GPU: RTX 4090 (24 GB) - xl-turbo (~9GB) + LM fit in 24GB.
R2 creds + ACESTEP_* via RunPod secrets/env.
"""
import json
import os
import subprocess
import time
import urllib.request
import urllib.error

import runpod

R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "music-generations")

SERVER_PORT = "8001"
_base = f"http://127.0.0.1:{SERVER_PORT}"

_server = None


def _post_json(path, payload, timeout=600):
    req = urllib.request.Request(
        _base + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ensure_server():
    """Boot the ACE-Step API server once (subprocess) and wait for /health."""
    global _server
    if _server is not None and _server.poll() is None:
        return
    env = {
        **os.environ,
        "ACESTEP_API_HOST": "0.0.0.0",
        "ACESTEP_API_PORT": SERVER_PORT,
        "ACESTEP_CONFIG_PATH": os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo"),
        "ACESTEP_LM_MODEL_PATH": os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-0.6B"),
        "ACESTEP_LM_BACKEND": os.getenv("ACESTEP_LM_BACKEND", "pt"),
        "ACESTEP_INIT_LLM": os.getenv("ACESTEP_INIT_LLM", "true"),
    }
    # workspace is where the repo was cloned + pip-installed
    cwd = os.getenv("ACESTEP_WORKDIR", "/workspace")
    log = open("/tmp/acestep_server.log", "a")
    _server = subprocess.Popen(
        ["python", "-m", "acestep.api_server"],
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    # wait for /health (model load can take minutes on cold start)
    deadline = time.time() + 3600
    while time.time() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(
                "ACE-Step server died on boot; log tail:\n"
                + open("/tmp/acestep_server.log").read()[-800:])
        try:
            with urllib.request.urlopen(_base + "/health", timeout=5) as r:
                if r.status == 200:
                    print("[worker] ACE-Step server healthy", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server boot timeout")


def _find_audio(obj, depth=0):
    if depth > 8 or obj is None:
        return None
    if isinstance(obj, str) and (
        obj.lower().endswith((".wav", ".mp3")) or obj.startswith(("/", "http"))):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            f = _find_audio(v, depth + 1)
            if f:
                return f
    if isinstance(obj, (list, tuple)):
        for v in obj:
            f = _find_audio(v, depth + 1)
            if f:
                return f
    return None


def upload_r2(data: bytes, key: str) -> str:
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_AKID,
        aws_secret_access_key=R2_SAK,
        region_name="auto",
    )
    s3.put_object(Bucket=R2_BUCKET, Key=key, Body=data, ContentType="audio/wav")
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": R2_BUCKET, "Key": key}, ExpiresIn=86400)


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
    r2_key = f"{session}/{out}" if session else out

    try:
        ensure_server()
        env_app = _post_json("/release_task", {
            "prompt": prompt,
            "lyrics": lyrics or "",
            "audio_duration": duration,
            "guidance_scale": 7.0,
        })
        data = env_app.get("data", env_app)
        task_id = data.get("task_id") or data.get("id") or env_app.get("task_id")
        if not task_id:
            return {"status": "error", "error": f"no task_id: {env_app}"[:500]}

        # poll
        result = None
        deadline = time.time() + 1500
        while time.time() < deadline:
            time.sleep(6)
            qr = _post_json("/query_result", {"task_id_list": [task_id]})
            qd = qr.get("data", qr)
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
                return {"status": "error", "error": str(task.get("error"))[:500]}

        if result is None:
            return {"status": "error", "error": "generation timeout"}

        audio_ref = _find_audio(result)
        if not audio_ref:
            return {"status": "error", "error": f"no audio in result: {list(result)[:8]}"}

        if audio_ref.startswith("http"):
            wav = urllib.request.urlopen(audio_ref, timeout=600).read()
        else:
            with urllib.request.urlopen(
                    _base + "/v1/audio?path=" + urllib.request.quote(audio_ref),
                    timeout=600) as r:
                wav = r.read()

        url = upload_r2(wav, r2_key)
        return {"status": "success", "r2_key": r2_key, "size": len(wav),
                "download_url": url}
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {e}"[:800]}


runpod.serverless.start({"handler": handler})