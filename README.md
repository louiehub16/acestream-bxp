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
| `dispatcher/dispatch.py` | Local CLI: `validate / enqueue / up / watch / down / logs / run` |
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
MAX_RUNTIME_SECONDS=21600`. Image bakes: `ACESTEP_CONFIG_PATH=acestep-v15-xl-turbo`
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
