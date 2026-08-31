#!/usr/bin/env bash
set -euo pipefail
# ---------------------------------------------------------------------------
# runpod/deploy_runpod.sh — ACE-Step xl-turbo bulk music on RunPod RTX 4090
#
# GETS EVERYTHING READY + runs the bulk once you say go. Steps:
#   1) Build the runpod worker image + push to Docker Hub (via GH Actions, or
#      local docker if present).
#   2) Create the RunPod serverless endpoint (RTX 4090, QUEUE type).
#   3) Submit the bulk CSV to the endpoint (each row = one song job).
#   4) Poll until done; download links are R2 presigned URLs.
#
# PREREQS (done for you): RunPod key at ~/.runpod_key, .env has R2 creds +
# DOCKERHUB_USER. Endpoint name via RUNPOD_ENDPOINT (default acestep-4090).
#
# NOTE: 'getting ready' = run `bash runpod/deploy_runpod.sh --prep` to build
# the image + create the endpoint WITHOUT submitting jobs. Run full to bulk.
# ---------------------------------------------------------------------------
RUNPOD_KEY="$(cat "$HOME/.runpod_key" | tr -d '[:space:]')"
export DOCKERHUB_USER="${DOCKERHUB_USER:-hrm3478938}"
ENDPOINT="${RUNPOD_ENDPOINT:-acestep-4090}"
IMAGE="${DOCKERHUB_USER}/acestep-runpod:latest"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

RED='\033[0;31m'; GRN='\033[0;32m'; NC='\033[0m'
say(){ echo -e "${GRN}[runpod]${NC} $*"; }
die(){ echo -e "${RED}[runpod]${NC} $*" >&2; exit 1; }

prep_only="${1:-full}"

# --- 0. validate key + R2 creds -----------------------------------------------
curl -s -m 20 "https://api.runpod.io/graphql" -H "Authorization: Bearer $RUNPOD_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"{ myself { id } }"}' | grep -q "user_" \
  && say "RunPod key valid" || die "RunPod key invalid"
[ -n "${R2_ENDPOINT_URL:-}" ] || { set -a; . "$REPO_DIR/.env"; set +a; }
: "${R2_ENDPOINT_URL:?R2_ENDPOINT_URL not in .env}"

# --- 1. build + push image ------------------------------------------------------
if command -v docker >/dev/null 2>&1; then
  say "building $IMAGE (uses cached layers if present)..."
  docker build -f "$REPO_DIR/runpod/Dockerfile" -t "$IMAGE" "$REPO_DIR"
  say "pushing $IMAGE..."
  docker push "$IMAGE"
else
  say "no local docker; triggering GH Actions build (docker-build.yml pushes acestep-runpod)..."
  # rely on GH Actions workflow (not written yet fallback) - dispatch note
  die "No local docker. Add a GH Actions workflow pushing hrm3478938/acestep-runpod:latest then rerun."
fi

# --- 2. create the serverless endpoint (RTX 4090 / ADA_24) ----------------------
say "creating RunPod endpoint '$ENDPOINT' (RTX 4090)..."
# Build the create payload with Python so env vars (R2 creds) are injected exactly.
# Docs (endpoint-configurations): queue type, gpu.pools (pool IDs), scaling
# REQUEST_COUNT, disk, container env vars.
python3 - "$REPO_DIR/.env" "$ENDPOINT" "$IMAGE" <<'PY' > /tmp/runpod_endpoint.json
import json, os, sys
env_path, endpoint, image = sys.argv[1], sys.argv[2], sys.argv[3]
# load .env
e = {}
for line in open(env_path, encoding="utf-8"):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("="); e[k.strip()] = v.strip()
# the env the worker + ACE-Step need
container_env = {
    "R2_ENDPOINT_URL": e.get("R2_ENDPOINT_URL", ""),
    "R2_ACCESS_KEY_ID": e.get("R2_ACCESS_KEY_ID", ""),
    "R2_SECRET_ACCESS_KEY": e.get("R2_SECRET_ACCESS_KEY", ""),
    "R2_BUCKET_NAME": e.get("R2_BUCKET_NAME", "music-generations"),
    "ACESTEP_CONFIG_PATH": "acestep-v15-xl-turbo",
    "ACESTEP_LM_MODEL_PATH": "acestep-5Hz-lm-0.6B",
    "ACESTEP_LM_BACKEND": "pt",
    "ACESTEP_INIT_LLM": "true",
    "RUNPOD_INIT_TIMEOUT": "800",
}
payload = {
    "name": endpoint,
    "type": "QUEUE",
    "image": image,
    "gpu": {"pools": ["ADA_24"]},          # RTX 4090 / RTX 6000 Ada
    "scaling": {"type": "REQUEST_COUNT", "requestCount": 1},
    "disk": 60,
    "env": container_env,                  # passed to the worker container
}
print(json.dumps(payload))
PY
ENDPOINT_ID="$(python3 -c "import json;print(json.load(open('/tmp/runpod_endpoint.json')).get('id',''))")"
[ -n "$ENDPOINT_ID" ] || { echo "endpoint create failed:"; cat /tmp/runpod_endpoint.json; exit 1; }
say "endpoint id: $ENDPOINT_ID"

# scale up max workers for bulk (RTX 4090 pool)
curl -s -m 30 -X POST "$ENDPOINT_ID" \
  -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
  -d '{"maxWorkers":8}' >/dev/null 2>&1 || say "(scaling tweak optional)"

[ "$prep_only" = "--prep" ] && { say "PREP DONE (image built+endpoint ready). Run full to submit jobs."; exit 0; }

# --- 3. submit bulk CSV -----------------------------------------------------------
CSV="${2:-$REPO_DIR/prompts.csv}"
: "${CSV:?usage: deploy_runpod.sh [--prep] <csrf.csv>}"
say "submitting jobs from $CSV to endpoint $ENDPOINT_ID..."
python3 "$REPO_DIR/runpod/submit_batch.py" "$ENDPOINT_ID" "$CSV"

say "DONE - watch outputs in R2 bucket (music-generations)."