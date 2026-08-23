#!/usr/bin/env python3
"""
ACE-Stream 5090 BXP - node sidecar.

Runs INSIDE the Salad container next to the official ACE-Step 1.5 API server.
Pull-model worker (nodes are outbound-only):

    loop:
      wait for ACE-Step server (/health)
      claim one job        GET  {QUEUE_BASE_URL}/claim?worker_id=<NODE_NAME>
      render               POST localhost/release_task -> /query_result -> /v1/audio
      upload               R2  <session>/<file>.wav  +  marker <session>/_done/<file>.json
      report               POST {QUEUE_BASE_URL}/complete {"job_id","ok",...}

Guarantees:
  - A claimed job is NEVER stranded: /complete always fires (ok true or false).
  - Deadman switch: MAX_RUNTIME_SECONDS hard exit so a frozen node stops billing
    (Salad restarts the container; the queue reclaims stale leases after 30 min).
  - Secrets never logged (masked to last 4 chars).
"""

import json
import os
import socket
import sys
import threading
import time

import boto3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------- env
QUEUE_BASE_URL = os.getenv("QUEUE_BASE_URL", "").rstrip("/")
ADMIN_KEY = os.getenv("ADMIN_KEY", "")
R2_ENDPOINT = os.getenv("R2_ENDPOINT_URL", "")
R2_AKID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SAK = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET_NAME", "")
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "900"))
MAX_RUNTIME = int(os.getenv("MAX_RUNTIME_SECONDS", "21600"))
API_PORT = os.getenv("ACESTEP_API_PORT", "8001")
NODE_NAME = os.getenv("NODE_NAME") or socket.gethostname()

API_BASE = f"http://127.0.0.1:{API_PORT}"
FALLBACK_PROMPT = "Cinematic industrial background layer"

log_lock = threading.Lock()
STOP_EVENT = threading.Event()


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] [{NODE_NAME}] {msg}"
    with log_lock:
        print(line, flush=True)


def mask(secret: str) -> str:
    return ("***" + secret[-4:]) if len(secret) >= 4 else ("***" if secret else "<unset>")


# ---------------------------------------------------------------- http
def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=2,
                  status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = make_session()
CLAIM_SESSION = requests.Session()


def queue_headers():
    return {"X-Admin-Key": ADMIN_KEY, "Content-Type": "application/json"}


def claim_job():
    """Ask the queue for one job. Returns dict or None."""
    r = CLAIM_SESSION.get(f"{QUEUE_BASE_URL}/claim",
                    params={"worker_id": NODE_NAME},
                    headers=queue_headers(), timeout=30)
    r.raise_for_status()
    return (r.json() or {}).get("job")


def report_complete(job_id: str, ok: bool, *, r2_key=None, error=None):
    body = {"job_id": job_id, "ok": ok}
    if ok:
        body["r2_key"] = r2_key
    else:
        body["error"] = (error or "unknown")[:500]
    last = None
    for attempt in range(5):
        try:
            r = SESSION.post(f"{QUEUE_BASE_URL}/complete",
                             headers=queue_headers(), data=json.dumps(body), timeout=30)
            r.raise_for_status()
            log(f"job {job_id}: reported ok={ok} (attempt {attempt + 1})")
            return
        except Exception as e:  # noqa: BLE001 - retry, never let reporting kill the loop
            last = e
            time.sleep(10)
    log(f"job {job_id}: ERROR complete-report failed after 5 tries: {last}")


# ---------------------------------------------------------------- r2
_r2 = None


def r2_client():
    global _r2
    if _r2 is None:
        _r2 = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_AKID, aws_secret_access_key=R2_SAK,
            region_name="auto",
        )
    return _r2


def sanitize_filename(name: str) -> str:
    base = os.path.basename((name or "").strip().replace("\\", "/"))
    safe = "".join(c for c in base if c.isalnum() or c in "._-") or "track.wav"
    return safe if safe.lower().endswith(".wav") else safe + ".wav"


def sanitize_session(s):
    base = (s or "").strip().replace("\\", "/").split("/")[-1]
    return "".join(c for c in base if c.isalnum() or c in "._-") or "misc"


# ---------------------------------------------------------------- acestep api
def wait_for_server(deadline_ts: float):
    log(f"waiting for ACE-Step server at {API_BASE} ...")
    announced = False
    while time.time() < deadline_ts:
        try:
            if SESSION.get(f"{API_BASE}/health", timeout=10).status_code == 200:
                if not announced:
                    log("ACE-Step server healthy")
                    announced = True
                return True
        except Exception:  # noqa: BLE001 - server may be mid-download of weights
            pass
        time.sleep(10)
    return False


def find_audio_path(obj, depth=0):
    """Recursively hunt for the first audio path/url in an arbitrary result dict."""
    if depth > 8 or obj is None:
        return None
    if isinstance(obj, str):
        low = obj.lower()
        if low.startswith("http") or low.endswith((".wav", ".mp3")):
            return obj
        return None
    if isinstance(obj, dict):
        for key in ("audio_path", "output_path", "path", "url", "audio_url", "file"):
            if isinstance(obj.get(key), str):
                found = find_audio_path(obj[key], depth + 1)
                if found:
                    return found
    if isinstance(obj, dict):
        for v in obj.values():
            found = find_audio_path(v, depth + 1)
            if found:
                return found
    if isinstance(obj, (list, tuple)):
        for v in obj:
            found = find_audio_path(v, depth + 1)
            if found:
                return found
    return None


def render(prompt: str, lyrics: str, duration_s: int) -> bytes:
    """Submit one generation task and return raw WAV bytes."""
    payload = {
        "prompt": prompt or FALLBACK_PROMPT,
        "lyrics": lyrics or "",
        "audio_duration": int(duration_s),
    }
    t0 = time.time()
    resp = SESSION.post(f"{API_BASE}/release_task",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    envelope = resp.json()
    data = envelope.get("data", envelope) if isinstance(envelope, dict) else {}
    task_id = data.get("task_id") or data.get("id") or envelope.get("task_id")
    if not task_id:
        raise RuntimeError(f"no task_id in release_task response: {str(envelope)[:300]}")

    # poll /query_result until status 1 (done) or 2 (failed); 0 = queued/running
    deadline = time.time() + JOB_TIMEOUT
    result = None
    while time.time() < deadline:
        time.sleep(5)
        qr = SESSION.post(f"{API_BASE}/query_result",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"task_id_list": [task_id]}), timeout=60)
        qr.raise_for_status()
        qdata = qr.json().get("data", qr.json())
        tasks = qdata.get("tasks") if isinstance(qdata, dict) else None
        task = (tasks[0] if tasks else qdata.get(task_id)) or (
            qdata[0] if isinstance(qdata, list) and qdata else None)
        if task is None:
            continue
        status = task.get("status")
        if status == 1:
            result = task
            break
        if status == 2:
            raise RuntimeError(f"task failed: {str(task.get('error'))[:300]}")
    if result is None:
        raise TimeoutError(f"task {task_id} exceeded {JOB_TIMEOUT}s")

    audio_ref = find_audio_path(result)
    if not audio_ref:
        raise RuntimeError(f"no audio path in result keys: {list(result)[:20]}")

    # fetch bytes: server route first, direct URL fallback
    if audio_ref.startswith("http"):
        r = SESSION.get(audio_ref, timeout=600)
        r.raise_for_status()
        wav = r.content
    else:
        dl = SESSION.get(f"{API_BASE}/v1/audio", params={"path": audio_ref}, timeout=600)
        if dl.status_code == 404 and audio_ref.startswith("/"):
            dl = SESSION.get(f"{API_BASE}/v1/audio",
                             params={"path": audio_ref.lstrip("/")}, timeout=600)
        dl.raise_for_status()
        wav = dl.content

    log(f"rendered in {time.time() - t0:.1f}s ({len(wav)} bytes)")
    return wav


# ---------------------------------------------------------------- job runner
def run_job(job: dict) -> None:
    try:
        job_id = str(job.get("id", "?"))
        session = sanitize_session(job.get("session", "misc"))
        filename = sanitize_filename(job.get("output_filename", ""))
        try:
            duration = int(job.get("duration") or 240)
        except ValueError:
            report_complete(job_id, False, error="bad duration")
            return
        duration = max(10, min(600, duration))
        r2_key = f"{session}/{filename}"
        wav = render(job.get("prompt", ""), job.get("lyrics", ""), duration)
        r2_client().put_object(Bucket=R2_BUCKET, Key=r2_key,
                               Body=wav, ContentType="audio/wav")
        r2_client().put_object(
            Bucket=R2_BUCKET, Key=f"{session}/_done/{filename}.json",
            Body=json.dumps({"job_id": job_id, "node": NODE_NAME,
                             "r2_key": r2_key, "ts": time.time(),
                             "bytes": len(wav)}),
            ContentType="application/json")
        report_complete(job_id, True, r2_key=r2_key)
    except Exception as e:  # noqa: BLE001 - report and let queue retry
        report_complete(job_id, False, error=f"{type(e).__name__}: {e}")


def deadman():
    time.sleep(max(60, MAX_RUNTIME - 120))
    STOP_EVENT.set()
    log("STOP_EVENT set - draining")
    time.sleep(max(180, JOB_TIMEOUT + 60))
    log("DEADMAN hard exit")
    os._exit(3)


def main() -> int:
    log(f"sidecar boot | node={NODE_NAME} queue={mask(QUEUE_BASE_URL) or '<unset>'} "
        f"bucket={mask(R2_BUCKET) or '<unset>'} r2={mask(R2_ENDPOINT) or '<unset>'} "
        f"admin_key={mask(ADMIN_KEY)} job_timeout={JOB_TIMEOUT}s max_runtime={MAX_RUNTIME}s")

    if not QUEUE_BASE_URL:
        log("QUEUE_BASE_URL unset - queue disabled, entering idle mode "
            "(container stays up for debugging)")
        while True:
            time.sleep(3600)

    if not (R2_ENDPOINT and R2_AKID and R2_SAK and R2_BUCKET):
        log("FATAL: R2 env incomplete - cannot upload results")
        return 2

    ok = wait_for_server(time.time() + 3600)
    if not ok:
        log("FATAL: ACE-Step server never became healthy")
        return 2
    log("server ready - starting poll loop")

    threading.Thread(target=deadman, daemon=True).start()
    log(f"polling loop start (deadman armed at +{MAX_RUNTIME}s)")

    while True:
        if STOP_EVENT.is_set():
            log("shutdown: stop claiming")
            break
        try:
            job = claim_job()
        except Exception as e:  # noqa: BLE001 - transient queue errors
            log(f"claim failed ({e}); retrying in 15s")
            time.sleep(15)
            continue

        if job is not None and not isinstance(job, dict):
            log("malformed job payload, skipping")
            time.sleep(15)
            continue
        elif job is None:
            time.sleep(15)
            continue

        log(f"claimed job {job.get('id')} -> {job.get('session')}/{job.get('output_filename')}")
        run_job(job)


if __name__ == "__main__":
    sys.exit(main())
