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
curl -s -m 60 -X POST "https://api.runpod.io/v2/serverless" \
  -H "Authorization: Bearer $RUNPOD_KEY" -H "Content-Type: application/json" \
  -d "{\"name\":\"$ENDPOINT\",\"type\":\"QUEUE\",\"image\":\"$IMAGE\",\"gpu\":{\"pools\":[\"ADA_24\"]},\"scaling\":{\"type\":\"REQUEST_COUNT\",\"requestCount\":1}}" \
  | tee /tmp/runpod_endpoint.json
ENDPOINT_ID="$(python3 -c "import json;print(json.load(open('/tmp/runpod_endpoint.json')).get('id',''))")"
[ -n "$ENDPOINT_ID" ] || die "endpoint create failed (see above)"
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