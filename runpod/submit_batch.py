#!/usr/bin/env python3
"""submit_batch.py — read the CSV job sheet and submit each row to the RunPod
serverless endpoint. Each row = one song job. Prints progress + final R2 summary.

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

RUNPOD_API = "https://api.runpod.io/v2/{endpoint}/runsync"


def submit(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        RUNPOD_API.format(endpoint=endpoint),
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


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
    endpoint, csv_path = sys.argv[1], sys.argv[2]
    jobs = load_jobs(csv_path)
    print(f"[submit] {len(jobs)} jobs -> endpoint {endpoint}")
    ok = 0
    for j in jobs:
        try:
            out = submit(endpoint, j)
            st = out.get("output") or out.get("status")
            ok += 1
            print(f"  ok {j['id']} {j['input']['output_filename']} -> {st}")
        except urllib.error.HTTPError as e:
            print(f"  FAIL {j['id']}: HTTP {e.code} {e.read()[:200]}")
    print(f"[submit] done: {ok}/{len(jobs)} submitted")


if __name__ == "__main__":
    main()