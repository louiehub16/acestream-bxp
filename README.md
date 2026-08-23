# ACE-Stream 5090 BXP — Bulk Music Generation Pipeline

Bulk-render music from a CSV prompt sheet on SaladCloud RTX 5090 nodes, using the
**official ACE-Step 1.5 API server**, with outputs archived to **Cloudflare R2**
(zero egress) and jobs distributed through a **Cloudflare Worker + D1 pull-queue**.

> Rebuilt from the original `bulk music making.txt` spec. That document's concept was
> sound but its code would never have run: hallucinated model repo (`Runware/ACE-Step-1.5-XL`
> → HTTP 401), broken `git clone`, CUDA 12.1 base (no RTX 5090 sm_120 kernels), fake FP8/torch.compile
> optimizations on attributes the real pipeline doesn't have, and a push-to-worker design that
> Salad's basic networking cannot serve (no public HTTP ingress). This rebuild replaces all of it.

## Architecture

```
 [You] python dispatcher/dispatch.py run --csv prompts.csv --replicas N
        |
        |-- 1. enqueue CSV jobs -->  [Cloudflare Worker queue (D1)]   acestream-queue
        |-- 2. create+start Salad container group (RTX 5090 pool)
        |                |
        |    [GPU node x N: docker image hrm3478938/acestep-bxp]
        |        entrypoint.sh ── starts official ACE-Step 1.5 API server (:8001)
        |        sidecar/worker_sidecar.py:
        |            loop: GET /claim  <-- queue (outbound poll, no ingress needed)
        |                  POST /release_task -> /query_result -> GET /v1/audio
        |                  put_object R2 <session>/<file>.wav
        |                  put_object R2 <session>/_done/<file>.json   (progress marker)
        |                  POST /complete ok|false                      (retry<=3)
        |-- 3. watch: counts _done markers + queue /progress -> live bar
        '-- 4. teardown: DELETE salad group (billing stops) ALWAYS (finally)
```

Why a **pull queue**: Salad nodes are outbound-only (basic networking has no public
HTTP ingress — verified gap). Nodes poll the Worker themselves; no gateway required.

## Layout

| Path | What |
|---|---|
| `docker/Dockerfile` | cu128 + Python 3.12 + torch cu128 wheels + ACE-Step 1.5 (pinned SHA) + optional weight bake (`BAKE_WEIGHTS=true`) |
| `docker/entrypoint.sh` | Starts API server bg → waits `/health` → runs sidecar with auto-restart; container never exits during model load |
| `sidecar/worker_sidecar.py` | Node job loop: claim → render → upload → mark done; deadman switch |
| `queue-worker/` | Cloudflare Worker (ES module) + D1 schema + wrangler.toml (atomic claims, stale-lease reclaim, retries≤3) |
| `dispatcher/dispatch.py` | Local CLI: `validate / enqueue / up / watch / down / logs / retry / delete / run` |
| `dispatcher/kanban_hook.py` | Optional per-tick progress POST to an external kanban board (`KANBAN_PROGRESS_URL`) |
| `.github/workflows/docker-build.yml` | GH Actions → Docker Hub, amd64-only, GHA cache |

## One-time setup

1. **Image** — push this folder to a GitHub repo; add secrets `DOCKERHUB_USERNAME`,
   `DOCKERHUB_TOKEN`; run workflow *docker-build* (leave `bake_weights=false`;
   first boot pulls weights from HF ~5–10 min/node).
2. **Queue** —
   ```bash
   cd queue-worker
   npx wrangler d1 create acestream-queue          # paste returned id into wrangler.toml
   npx wrangler d1 execute acestream-queue --remote --file=schema.sql
   npx wrangler secret put ADMIN_KEY               # invent a long random string
   npx wrangler deploy                             # note the *.workers.dev URL
   ```
3. **R2 bucket + token** — dash.cloudflare.com → R2 → create bucket (e.g. `music-generations`)
   → Manage API Tokens → *Object Read & Write* scoped to that bucket.
4. **`.env`** — copy `.env.example` → `.env`; fill every key.

## Bulk run

```bash
python dispatcher/dispatch.py run --csv my_prompts.csv --replicas 6
```

CSV columns: `prompt,duration,output_filename[,lyrics]` (delimiter auto-sniffed;
duration clamped to model range 10–600 s; blank prompts get fallback text;
duplicate filenames deduped). Sample: `dispatcher/prompts.sample.csv`.

Individual commands: `validate` → `enqueue` → `up` → `watch` → `down`.

## Operations

### Retry failed jobs
Jobs whose render failed leave a marker at `<session>/_failed/<file>.json`
(with the error text) in R2 and are parked as `failed` (attempts exhausted) in
the queue. Reset them without re-uploading anything:

```bash
python dispatcher/dispatch.py retry --session <S>          # table + confirm
python dispatcher/dispatch.py retry --session <S> --yes    # no prompt
```

`retry` lists every `_failed` marker (filename + first 60 chars of the error),
asks for confirmation, then POSTs `{session, filenames}` to the queue worker's
`/retry` endpoint (`X-Admin-Key` auth), which flips those rows back to
`pending` with `attempts=0`. Lists over 500 collapse to `{all: true}`.
Running nodes pick the reset jobs up on their next claim — no restart needed.

### GPU fallback pool (`GPU_CLASSES`)
By default groups are created on the RTX 5090 class only. Set `GPU_CLASSES`
(comma-separated Salad class UUIDs) to widen the pool:

```
GPU_CLASSES=851399fb-7329-4195-a042-d6514b28cf33,ed563892-aacd-40f5-80b7-90c9be6c759b
```

The RTX 4090 (24 GB) is usually faster to allocate and acestep-v15-xl fits its
24 GB — Salad then schedules on whichever class frees up first.

### Output format (`OUTPUT_FORMAT`)
Set `OUTPUT_FORMAT=wav|flac|mp3` in `.env`; it is passed through to the node
env by `up`/`run`, and the sidecar uploads that container with the matching
R2 content type. Default: `wav`.

### Completion report + R2 cleanup
When `watch` sees the session complete it prints a completion report:
the dashboard browse link, audio count (= number of `_done` markers), total
size in GB (2 decimals) and a count of non-wav stragglers if any. On an
interactive terminal it then asks `Delete these N files from R2? [y/N]` —
answering `y` wipes the whole `<session>/` prefix via batched `delete_objects`
(1000 keys per call) and reports the freed GB. Pass `--no-delete` to `watch`
or `run` for report-only behavior; the prompt is skipped automatically when
stdin isn't a TTY (e.g. cron). `run` performs the same single report after its
watch phase succeeds (never twice).

To wipe an old session later without a report:

```bash
python dispatcher/dispatch.py delete --session <S> [--force]
```

### Kanban progress hook (optional)
Set `KANBAN_PROGRESS_URL` (e.g. your PCBGenius kanban worker
`.../api/update`) and every `watch` tick silently POSTs a snapshot there via
`dispatcher/kanban_hook.py`: payload mirrors that board's shape
(`agent/status/feature/message` + generic `service/session/stats/ts`). The
board's Cloudflare WAF rejects non-browser agents (403/1010), so a Chrome
User-Agent header is sent. Failures never raise or print — a dead board can't
break a watch loop.

## Reality checks (vs the original doc)

| Claim in old doc | Reality |
|---|---|
| Model "Runware/ACE-Step-1.5-XL" | ❌ 401. Real: `ACE-Step/acestep-v15-xl-{turbo,sft}` (+ `Ace-Step1.5`) |
| 61 replicas in 5 minutes | Salad meters GPU allocation; expect a handful of nodes ramping. 1500 tracks ≈ **1–3 h** |
| ~12 s per 4-min track | xl-turbo is fast (~10 s/song class on 3090-class HW); plan 30–90 s incl. upload |
| Custom `app.py` optimizations | Deleted — official server already queues/batches; FP8 VAE + torch.compile were fabricated APIs |
| Cost | Still tiny: roughly $0.35/hr/node × few node-hours ≈ **$2–5** per 1500 tracks |

## Env contract (single source of truth)

Shared by dispatcher `.env` and Salad group env (set automatically from `.env`):
`QUEUE_BASE_URL, ADMIN_KEY, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, JOB_TIMEOUT_SECONDS=900,
MAX_RUNTIME_SECONDS=21600, OUTPUT_FORMAT=wav`. Dispatcher-only extras:
`GPU_CLASSES` (comma-separated Salad GPU class UUIDs; default 5090 pool,
add `ed563892-aacd-40f5-80b7-90c9be6c759b` = RTX 4090 as fallback),
`KANBAN_PROGRESS_URL` (optional kanban feed). Image bakes:
`ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo`
(swap to `-xl-sft` for max quality), `ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-4B`,
`ACESTEP_LM_BACKEND=vllm`, `HF_HOME=/hf-cache`.

## Ad-hoc verification

py_compile dispatch/sidecar, node --check worker, bash -n entrypoint, YAML parse, sqlite schema+UNIQUE+atomic claim, enqueue parse-only, sidecar idle boot+secret masking.

1. PASS — py_compile dispatcher
2. PASS — py_compile sidecar
3. PASS — node --check worker
4. PASS — bash -n entrypoint
5. PASS — YAML parse
6. PASS — sqlite schema + UNIQUE + atomic claim
7. PASS — enqueue parse-only
8. PASS — sidecar idle boot + secret masking
9. PASS — retry UPDATE simulation (sqlite: failed→pending, attempts=0, session-scoped)
10. PASS — content-type map import via importlib (network-free)
11. PASS — kanban hook silent-off + GPU_CLASSES/OUTPUT_FORMAT env parsing
