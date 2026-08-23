#!/usr/bin/env python3
"""
ACE-Stream 5090 BXP - local dispatcher CLI.

Control plane for bulk music generation on SaladCloud RTX 5090 nodes:

    validate   check .env + queue + R2 + Salad auth
    enqueue    parse CSV -> push jobs to Cloudflare queue
    up         create + start the Salad container group (5090 pool)
    watch      live progress (R2 _done markers + queue stats)
    down       delete group -> billing stops
    logs       tail Salad system logs (event-level)
    run        full pipeline: validate->enqueue->up->watch->down

Requires: requests, boto3   (pip install requests boto3)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("missing dependency: pip install requests", file=sys.stderr)
    sys.exit(2)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

EXIT_OK, EXIT_CONFIG, EXIT_API = 0, 2, 3

SALAD_BASE = "https://api.salad.com/api/public"
SALAD_ORG = "lucas16"
SALAD_PROJECT = "default"
GPU_5090 = "851399fb-7329-4195-a042-d6514b28cf33"  # RTX 5090 (32 GB) in org lucas16
ACCOUNT_ID = "abd12cd58366e2d99a202218328b1340"
FALLBACK_PROMPT = "Cinematic industrial background layer"
DURATION_MIN, DURATION_MAX = 10, 600


# ---------------------------------------------------------------- .env
def load_env() -> dict:
    env = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip().strip('"').strip("'")
    return env


ENV = load_env()


def env_get(key: str, default: str = "") -> str:
    return os.environ.get(key) or ENV.get(key) or default


def r2_endpoint_defaulted() -> str:
    return env_get("R2_ENDPOINT_URL",
                   f"https://{ACCOUNT_ID}.r2.cloudflarestorage.com")


# ---------------------------------------------------------------- http helper
def retry_call(fn, tries: int = 3, backoff: float = 3.0, what: str = "call"):
    last = None
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < tries:
                print(f"  [{what}] attempt {attempt} failed ({e}); retrying...")
                time.sleep(backoff * attempt)
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------- salad api
def salad_headers():
    return {"Salad-Api-Key": env_get("SALAD_API_KEY"),
            "Content-Type": "application/json"}


def salad_create_group(name: str, image: str, replicas: int, node_env: dict,
                       dry_run: bool = False) -> dict:
    body = {
        "name": name,
        "container": {
            "image": image,
            "resources": {
                "cpu": 8,
                "memory": 32768,
                "gpu_classes": [GPU_5090],   # snake_case or GPUs silently vanish
                "shm_size": 64,
            },
            "networking": {"port": 8001, "auth": False, "protocol": "http"},
            "environment_variables": node_env,
        },
        "replicas": replicas,
    }
    if dry_run:
        return body

    url = f"{SALAD_BASE}/organizations/{SALAD_ORG}/projects/{SALAD_PROJECT}/containers"

    def do_create():
        r = requests.post(url, headers=salad_headers(), data=json.dumps(body), timeout=60)
        if r.status_code == 400 and "name_conflict" in r.text:
            raise NameConflict(r.text[:200])
        r.raise_for_status()
        return r

    try:
        retry_call(do_create, what="salad-create")
    except NameConflict:
        name = f"{name}-{uuid.uuid4().hex[:6]}"
        print(f"  name conflict; retrying as {name}")
        body["name"] = name

        def do_create2():
            r = requests.post(url, headers=salad_headers(),
                              data=json.dumps(body), timeout=60)
            r.raise_for_status()
            return r
        retry_call(do_create2, what="salad-create-retry")

    # autostart_policy defaults false -> explicit start with NO body (body => 400)
    start_url = f"{url}/{body['name']}/start"
    retry_call(lambda: requests.post(
                   start_url, headers=salad_headers(),
                   timeout=60).raise_for_status(),
               what="salad-start")
    return body


class NameConflict(Exception):
    pass


def salad_get_group(name: str):
    url = f"{SALAD_BASE}/organizations/{SALAD_ORG}/projects/{SALAD_PROJECT}/containers/{name}"
    r = requests.get(url, headers=salad_headers(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def salad_delete_group(name: str) -> bool:
    url = f"{SALAD_BASE}/organizations/{SALAD_ORG}/projects/{SALAD_PROJECT}/containers/{name}"
    r = requests.delete(url, headers=salad_headers(), timeout=60)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    return True


def salad_logs(name: str):
    url = (f"{SALAD_BASE}/organizations/{SALAD_ORG}/projects/"
           f"{SALAD_PROJECT}/containers/{name}/system-logs")
    r = requests.get(url, headers=salad_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- queue api
def queue_base() -> str:
    return env_get("QUEUE_BASE_URL").rstrip("/")


def queue_health() -> dict:
    r = retry_call(lambda: requests.get(f"{queue_base()}/health", timeout=30),
                   what="queue-health")
    r.raise_for_status()
    return r.json()


def queue_push_jobs(jobs: list[dict]) -> dict:
    headers = {"X-Admin-Key": env_get("ADMIN_KEY"),
               "Content-Type": "application/json"}

    def post(chunk):
        r = requests.post(f"{queue_base()}/jobs", headers=headers,
                          data=json.dumps({"jobs": chunk}), timeout=120)
        r.raise_for_status()
        return r.json()

    out = {"queued": 0, "duplicates": 0}
    for i in range(0, len(jobs), 500):
        chunk = jobs[i:i + 500]
        res = retry_call(lambda c=chunk: post(c), what=f"queue-jobs[{i}]")
        out["queued"] += res.get("queued", 0)
        out["duplicates"] += res.get("duplicates", 0)
    return out


def queue_progress(session: str) -> dict:
    headers = {"X-Admin-Key": env_get("ADMIN_KEY")}
    r = retry_call(lambda: requests.get(
        f"{queue_base()}/progress", params={"session": session},
        headers=headers, timeout=30), what="queue-progress")
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- r2 helpers
def r2_client():
    try:
        import boto3
    except ImportError:
        print("missing dependency: pip install boto3", file=sys.stderr)
        sys.exit(EXIT_CONFIG)
    return boto3.client(
        "s3", endpoint_url=r2_endpoint_defaulted(),
        aws_access_key_id=env_get("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=env_get("R2_SECRET_ACCESS_KEY"),
        region_name="auto")


def count_done_markers(bucket: str, session: str) -> tuple[int, list[str]]:
    c = r2_client()
    n, recent = 0, []
    paginator = c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{session}/_done/"):
        for obj in page.get("Contents", []):
            n += 1
            recent.append((obj.get("LastModified"), obj["Key"]))
    recent.sort(reverse=True)
    return n, [k.rsplit("/", 1)[-1].removesuffix(".json") for _, k in recent[:5]]


# ---------------------------------------------------------------- csv parsing
def sanitize_filename(name: str) -> str:
    base = os.path.basename((name or "").strip().replace("\\", "/"))
    safe = "".join(ch for ch in base if ch.isalnum() or ch in "._-")
    if not safe:
        safe = f"track_{uuid.uuid4().hex[:6]}.wav"
    return safe if safe.lower().endswith(".wav") else safe + ".wav"


def parse_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")  # strips BOM
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel  # fallback comma
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise SystemExit(f"{path}: empty CSV")
    fields = [f.strip().lower() for f in reader.fieldnames]
    required = {"prompt", "duration", "output_filename"}
    missing = required - set(fields)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    jobs, seen_names = [], set()
    for idx, row in enumerate(reader, start=1):
        vals = {(k.strip().lower() if k else k): v for k, v in row.items()}
        if all((vals.get(c) or "") == "" for c in
               ("prompt", "duration", "output_filename")):
            continue  # fully blank row
        prompt = (vals.get("prompt") or "").strip() or FALLBACK_PROMPT
        lyrics = (vals.get("lyrics") or "").strip()
        try:
            duration = int(float(vals.get("duration") or 240))
        except ValueError:
            duration = 240
        duration = max(DURATION_MIN, min(DURATION_MAX, duration))
        filename = sanitize_filename(vals.get("output_filename"))
        stem, ext = filename[:-4], filename[-4:]
        dedup = 2
        while filename in seen_names:
            filename = f"{stem}_{dedup}{ext}"
            dedup += 1
        seen_names.add(filename)
        jobs.append({
            "id": f"{idx:04d}",
            "session": "",  # filled by caller
            "prompt": prompt,
            "lyrics": lyrics,
            "duration": duration,
            "output_filename": filename,
        })
    return jobs


# ---------------------------------------------------------------- ui helpers
def bar(done: int, total: int, width: int = 40) -> str:
    pct = done / total if total else 0.0
    filled = int(pct * width)
    return "#" * filled + "." * (width - filled)


def table(rows: list[tuple]) -> None:
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))


def require_config(keys: list[str]) -> list[str]:
    problems = []
    for k in keys:
        val = env_get(k)
        placeholder = val.upper().startswith(("PLACEHOLDER", "YOUR_", "<"))
        if not val or placeholder:
            problems.append(k)
    return problems


# ---------------------------------------------------------------- commands
def cmd_validate(_args) -> int:
    failures = []

    need = ["SALAD_API_KEY", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET_NAME"]
    if not env_get("R2_ENDPOINT_URL"):
        pass  # defaulted from account id
    missing = require_config(need)
    if missing:
        failures.append(f".env missing/incomplete: {', '.join(missing)}")

    qb = queue_base()
    if qb:
        try:
            h = queue_health()
            print(f"queue      OK  {qb} -> {h}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"queue unreachable: {e}")
    else:
        print("queue      SKIP  QUEUE_BASE_URL not set (set after wrangler deploy)")

    try:
        bucket = env_get("R2_BUCKET_NAME")
        if bucket:
            r2_client().head_bucket(Bucket=bucket)
            print(f"r2         OK  bucket '{bucket}' reachable")
        else:
            failures.append("R2_BUCKET_NAME unset")
    except Exception as e:  # noqa: BLE001
        failures.append(f"R2 unreachable: {e}")

    try:
        r = requests.get(
            f"{SALAD_BASE}/organizations/{SALAD_ORG}/projects",
            headers=salad_headers(), timeout=60)
        r.raise_for_status()
        names = [p.get("name") for p in r.json()]
        print(f"salad      OK  auth valid; projects={names}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"Salad API auth failed: {e}")

    print()
    if failures:
        print("VALIDATE: FAILURES")
        for f in failures:
            print(f"  x {f}")
        return EXIT_CONFIG
    print("VALIDATE: ALL OK")
    return EXIT_OK


def cmd_enqueue(args) -> int:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"csv not found: {csv_path}", file=sys.stderr)
        return EXIT_CONFIG
    session = args.session or datetime.now(timezone.utc).strftime("generation_%Y-%m-%d_%H-%M-%S")
    jobs = parse_csv(csv_path)
    for j in jobs:
        j["session"] = session
    if not jobs:
        print("no valid rows", file=sys.stderr)
        return EXIT_CONFIG

    rows = [("rows", len(jobs)), ("session", session),
            ("sample", f"{jobs[0]['id']}:{jobs[0]['output_filename']} "
                       f"<- {jobs[0]['prompt'][:48]}..." if jobs else "-")]
    table(rows)
    if args.parse_only:
        print("\n--parse-only: first 3 would-be payloads:")
        for j in jobs[:3]:
            print(json.dumps(j))
        return EXIT_OK

    res = queue_push_jobs(jobs)
    print(f"queued={res['queued']} duplicates={res['duplicates']}")
    print(f"next: python {Path(__file__).name} watch --session {session}")
    return EXIT_OK


def build_node_env() -> dict:
    keys = ["QUEUE_BASE_URL", "ADMIN_KEY", "R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME",
            "JOB_TIMEOUT_SECONDS", "MAX_RUNTIME_SECONDS"]
    node_env = {}
    for k in keys:
        v = env_get(k)
        if k == "R2_ENDPOINT_URL" and not v:
            v = r2_endpoint_defaulted()
        if k == "JOB_TIMEOUT_SECONDS" and not v:
            v = "900"
        if k == "MAX_RUNTIME_SECONDS" and not v:
            v = "21600"
        if v:
            node_env[k] = v
    return node_env


def cmd_up(args) -> int:
    if "/" in args.image:
        image = args.image
    else:
        user = env_get("DOCKERHUB_USER")
        if not user:
            print("DOCKERHUB_USER not set in .env", file=sys.stderr)
            return EXIT_CONFIG
        tag = env_get("ACESTEP_IMAGE_TAG", "latest")
        image = f"{user}/acestep-bxp:{tag}"

    name = args.name or f"acestream-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    node_env = build_node_env()

    created = salad_create_group(name, image, args.replicas, node_env,
                                 dry_run=args.dry_run)
    if args.dry_run:
        print("--dry-run Salad create payload:")
        print(json.dumps(created, indent=2))
        return EXIT_OK

    name = created["name"]
    args.created_group = name
    print(f"group '{name}' created+start sent; waiting for running state...")
    deadline = time.time() + 15 * 60
    last = ""
    while time.time() < deadline:
        g = salad_get_group(name)
        if g is None:
            print("group vanished after create?!", file=sys.stderr)
            return EXIT_API
        status = (g.get("current_state") or {}).get("status", "?")
        counts = g.get("instance_status_counts") or {}
        line = f"{status} {counts}"
        if line != last:
            print(f"  {line}")
            last = line
        if status == "running":
            print(f"UP: {args.replicas} replica(s) requested; nodes will pull jobs outbound.")
            print("(note: basic Salad networking exposes no public HTTP; that's expected -")
            print(" workers poll the queue themselves)")
            return EXIT_OK
        time.sleep(20)
    print("timeout waiting for running state; check `logs` command", file=sys.stderr)
    return EXIT_API


def cmd_watch(args) -> int:
    bucket = env_get("R2_BUCKET_NAME")
    session = args.session
    if not session:
        sessions = sorted({j["session"] for j in []})  # placeholder; require explicit
        _ = sessions
        print("--session required (printed by enqueue)", file=sys.stderr)
        return EXIT_CONFIG
    total_hint = args.total or 0
    start = time.time()
    last_done = -1
    while True:
        done, recent = count_done_markers(bucket, session)
        try:
            qp = queue_progress(session)
        except Exception:  # noqa: BLE001
            qp = {}
        total = total_hint or qp.get("total") or done or 0
        failed = qp.get("failed", 0)
        pending = qp.get("pending", 0)
        claimed = qp.get("claimed", 0)
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 and done > 0 else 0
        eta = (total - done - failed) / rate if rate and total else 0
        print("=" * 62)
        print(f"[{bar(done, total)}] {done}/{total} ({done / total * 100 if total else 0:.1f}%)")
        print(f"  states: done={done} failed={failed} pending={pending} "
              f"claimed={claimed} | elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m")
        if recent:
            print(f"  latest: {', '.join(recent)}")
        if total and done + failed >= total:
            print("=" * 62)
            print(f"SESSION COMPLETE: ok={done} failed={failed}")
            print(browse_link(session))
            return EXIT_OK
        if args.timeout and elapsed > args.timeout:
            print("watch timeout reached; run `down` to stop billing", file=sys.stderr)
            return EXIT_API
        time.sleep(args.interval)


def browse_link(session: str) -> str:
    bucket = env_get("R2_BUCKET_NAME")
    return (f"https://dash.cloudflare.com/{ACCOUNT_ID}/r2/default/buckets/"
            f"{bucket}?prefix={session}%2F")


def cmd_down(args) -> int:
    if not args.force:
        ans = input(f"delete Salad group '{args.name}'? [y/N] ")
        if ans.strip().lower() != "y":
            print("aborted")
            return EXIT_OK
    try:
        existed = salad_delete_group(args.name)
    except Exception as e:  # noqa: BLE001
        print(f"delete failed: {e}", file=sys.stderr)
        return EXIT_API
    if not existed:
        print(f"group '{args.name}' already gone")
        return EXIT_OK
    deadline = time.time() + 120
    gone = False
    while time.time() < deadline:
        if salad_get_group(args.name) is None:
            gone = True
            break
        time.sleep(5)
    if not gone:
        print(f"group '{args.name}' still exists - billing may continue.",
              file=sys.stderr)
        return EXIT_API
    print(f"DOWN: group '{args.name}' deleted - billing stopped.")
    return EXIT_OK


def cmd_logs(args) -> int:
    try:
        data = salad_logs(args.name)
    except Exception as e:  # noqa: BLE001
        print(f"log fetch failed: {e}", file=sys.stderr)
        return EXIT_API
    entries = data if isinstance(data, list) else data.get("logs", [])
    print(f"(event-level only; container stdout is NOT streamed by this API)")
    for e in entries[-args.lines:]:
        ts = e.get("log_entry_start", e.get("logged_at", ""))
        msg = e.get("entry", e.get("message", ""))
        print(f"{ts} | {msg}")


def cmd_run(args) -> int:
    t0 = time.time()
    session = datetime.now(timezone.utc).strftime("generation_%Y-%m-%d_%H-%M-%S")
    group_name = None
    rc_validate = cmd_validate(argparse.Namespace())
    if rc_validate != EXIT_OK and not args.force_continue:
        return rc_validate
    try:
        argv = ["enqueue", "--csv", args.csv, "--session", session]
        ns = build_parser().parse_args(argv)
        rc = ns.func(ns)
        if rc != EXIT_OK:
            return rc

        argv = ["up", "--replicas", str(args.replicas)] + (
            ["--dry-run"] if args.dry_run else [])
        ns = build_parser().parse_args(argv)
        rc = ns.func(ns)
        # grab the final (possibly conflict-suffixed) name returned by
        # salad_create_group even on failure, so teardown still runs
        group_name = getattr(ns, "created_group", None)
        if rc != EXIT_OK:
            return rc
        if args.dry_run:
            return EXIT_OK

        argv = ["watch", "--session", session, "--total", str(args.total or 0)]
        ns = build_parser().parse_args(argv)
        rc = ns.func(ns)
        if rc != EXIT_OK:
            return rc
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if group_name and not args.keep_group:
            print(f"teardown: deleting {group_name}")
            try:
                salad_delete_group(group_name)
                time.sleep(5)
                if salad_get_group(group_name) is None:
                    print("billing stopped.")
            except Exception as e:  # noqa: BLE001
                print(f"WARN teardown failed: {e} - DELETE IT MANUALLY", file=sys.stderr)
    hours = (time.time() - t0) / 3600
    est = args.replicas * hours * 0.35
    print("=" * 62)
    print(f"GRAND SUMMARY | wall={(time.time()-t0)/60:.1f}m | est cost ~${est:.2f}"
          f" ({args.replicas} replicas x $0.35/hr)")
    print(browse_link(session))
    return EXIT_OK


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dispatch.py",
                                description="ACE-Stream 5090 BXP dispatcher")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("validate", help="check .env, queue, R2, Salad auth")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("enqueue", help="push CSV rows to the job queue")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--session")
    sp.add_argument("--parse-only", action="store_true",
                    help="validate CSV and print payloads without network")
    sp.set_defaults(func=cmd_enqueue)

    sp = sub.add_parser("up", help="create + start Salad GPU group")
    sp.add_argument("--replicas", type=int, default=1)
    sp.add_argument("--image", default="")
    sp.add_argument("--name")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("watch", help="live progress until session completes")
    sp.add_argument("--session", required=True)
    sp.add_argument("--interval", type=int, default=30)
    sp.add_argument("--timeout", type=int, default=14400)
    sp.add_argument("--total", type=int, default=0, help="override total if queue unreachable")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("down", help="delete group (stop billing)")
    sp.add_argument("--name", required=True)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_down)

    sp = sub.add_parser("logs", help="tail Salad system logs (event-level)")
    sp.add_argument("--name", required=True)
    sp.add_argument("--lines", type=int, default=100)
    sp.set_defaults(func=cmd_logs)

    sp = sub.add_parser("run", help="full pipeline: validate>enqueue>up>watch>down")
    sp.add_argument("--csv", required=True)
    sp.add_argument("--replicas", type=int, default=1)
    sp.add_argument("--total", type=int, default=0)
    sp.add_argument("--keep-group", action="store_true")
    sp.add_argument("--force-continue", action="store_true")
    sp.add_argument("--dry-run", action="store_true")
    sp.set_defaults(func=cmd_run)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        sys.exit(args.func(args))
    except KeyboardInterrupt:
        print("\nctrl-c")
        sys.exit(EXIT_OK)
