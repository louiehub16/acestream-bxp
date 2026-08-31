#!/usr/bin/env python3
"""submit_batch.py — read the CSV job sheet and submit each row to a RunPod
serverless endpoint. Each row = one song job. Uses async /run + polling (NOT
runsync) so a cold-start model load (up to RUNPOD_INIT_TIMEOUT) is covered by
the per-job execution timeout, not a sync HTTP cap. Prints + writes results.

Docs: docs.runpod.io/serverless/endpoints/send-requests (execution timeout in
job policy).

Usage: python runpod/submit_batch.py <endpoint_id> <prompts.csv>
"""
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error

KEY = open(os.path.expanduser("~/.runpod_key")).read().strip()

BASE = "https://api.runpod.ai/v2/{eid}"
# Per-job execution timeout: 30 min (ms) — covers first cold-start model load.
EXEC_TIMEOUT_MS = 30 * 60 * 1000


def call(eid: str, path: str, method="POST", payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{BASE.format(eid=eid)}/{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def submit(eid: str, job: dict) -> str:
    """Submit one job with an execution-timeout policy; return job id."""
    body = {"input": job["input"], "executionTimeout": EXEC_TIMEOUT_MS}
    out = call(eid, "run", payload=body)
    return out.get("id")


def poll(eid: str, job_id: str, timeout_s=1800):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = call(eid, f"status/{job_id}", method="GET", timeout=60)
        s = st.get("status")
        if s in ("COMPLETED", "FAILED", "CANCELLED"):
            return st
        time.sleep(10)
    return {"status": "POLL_TIMEOUT", "error": f"timed out watching {job_id}"}


def load_jobs(csv_path: str) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise SystemExit(f"{csv_path}: empty CSV")
        fields = [x.strip().lower() for x in reader.fieldnames]
        need = {"prompt", "duration", "output_filename"}
        missing = need - set(fields)
        if missing:
            raise SystemExit(f"CSV missing required columns: {sorted(missing)}")
        jobs = []
        for i, row in enumerate(reader, 1):
            vals = {(k or "").strip().lower() if k else k: v for k, v in row.items()}
            prompt = (vals.get("prompt") or "").strip() or "Cinematic industrial background layer"
            lyrics = (vals.get("lyrics") or "").strip()
            try:
                duration = int(float(vals.get("duration") or 240))
            except ValueError:
                duration = 240
            duration = max(10, min(600, duration))
            jobs.append({
                "id": f"{i:04d}",
                "input": {
                    "prompt": prompt,
                    "lyrics": lyrics,
                    "duration": duration,
                    "output_filename": vals.get("output_filename") or f"track_{i}.wav",
                },
            })
    return jobs


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    eid, csv_path = sys.argv[1], sys.argv[2]
    jobs = load_jobs(csv_path)
    print(f"[submit] {len(jobs)} jobs -> endpoint {eid}")
    results = []
    for j in jobs:
        try:
            jid = submit(eid, j)
            print(f"  submitted {j['id']} -> {jid}")
            st = poll(eid, jid)
            out = st.get("output")
            status = st.get("status")
            ok = status == "COMPLETED"
            print(f"  {j['id']} {status}: {json.dumps(out)[:200] if out else st.get('error')}")
            results.append({"id": j["id"], "status": status, **({"output": out} if ok else {"error": st.get("error")})})
        except (urllib.error.HTTPError) as e:
            print(f"  FAIL {j['id']}: HTTP {e.code} {e.read()[:200]}")
    out_path = sys.argv[3] if len(sys.argv) > 3 else "runpod_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[submit] wrote {out_path}; {sum(1 for r in results if r.get('status')=='COMPLETED')}/{len(jobs)} complete")


if __name__ == "__main__":
    main()