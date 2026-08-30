#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# ACE-Stream 5090 BXP - container entrypoint
#
# WHY THIS SHAPE: Salad kills containers whose main process exits (restart
# loop "Instance Exited:1"). Model weights download + load takes minutes on
# first boot, so the API server must start in the BACKGROUND while this
# script (PID 1) stays alive forever. The sidecar only starts once /health
# answers, and it auto-restarts if it ever crashes. Nothing here exits.
# ---------------------------------------------------------------------------
set -uo pipefail

PORT="${ACESTEP_API_PORT:-8001}"
APP_DIR="/app/ACE-Step-1.5"
SIDECAR="/app/sidecar/worker_sidecar.py"
LOG="/tmp/acestep.log"

echo "[entrypoint] boot | host=$(hostname) | api_port=${PORT}"

# --- optional SSH debug server (only if SSH_PUBLIC_KEY provided) -------------
if [[ -n "${SSH_PUBLIC_KEY:-}" ]]; then
  echo "[entrypoint] installing sshd (debug mode)"
  mkdir -p /run/sshd /root/.ssh
  echo "${SSH_PUBLIC_KEY}" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq openssh-server >/dev/null 2>&1 || true
  /usr/sbin/sshd -p 2222 2>/dev/null \
    && echo "[entrypoint] sshd up on :2222" \
    || echo "[entrypoint] WARNING sshd failed (non-fatal)"
fi

# --- start ACE-Step API server in background ----------------------------------
cd "${APP_DIR}"
python -m acestep.api_server --host "${ACESTEP_API_HOST:-::}" --port "${PORT}" >>"${LOG}" 2>&1 &
SERVER_PID=$!
echo "[entrypoint] ACE-Step API server starting (pid ${SERVER_PID}), logs -> ${LOG}"

# --- readiness gate ------------------------------------------------------------
HEALTHY_ONCE=0
while true; do
  if ! curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    sleep 5
    continue
  fi
  if [[ "${HEALTHY_ONCE}" -eq 0 ]]; then
    echo "[entrypoint] /health OK - ACE-Step server ready"
    HEALTHY_ONCE=1
  fi
  break
done

# --- server watchdog (log-only; never exits the container) ---------------------
(
  while true; do
    sleep 30
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[entrypoint] WARNING: API server died post-startup - see ${LOG}; restarting"
      tail -n 80 "${LOG}" || true
      python -m acestep.api_server --host "${ACESTEP_API_HOST:-::}" --port "${PORT}" >>"${LOG}" 2>&1 &
      SERVER_PID=$!
    fi
  done
) &

# --- OBSERVABILITY: heartbeat + log tail -> R2 (debug the 'alive but stuck' gap)
# Ships the ACE-Step server's real last log lines to R2 every 30s so the outside
# watcher sees exactly where model-load/generation is - no SSH needed. Uses the
# venv boto3 (PATH already points there). Non-fatal: never brings the container down.
if [[ -n "${R2_BUCKET_NAME:-}" && -n "${R2_ENDPOINT_URL:-}" ]]; then
  HB_KEY="state/heartbeat_${NODE_NAME:-$(hostname)}.txt"
  (
    while true; do
      sleep 30
      PLOG="$(tail -c 2000 "${LOG}" 2>/dev/null | tr '\n' '|' | tail -c 2000)"
      HOST="$(hostname)"
      printf 'host=%s ts=%s alive | PLOG: %s' \
        "${HOST}" "$(date -u +%FT%TZ)" "${PLOG}" | \
        python -c "import boto3,sys,os; boto3.client('s3', endpoint_url=os.environ['R2_ENDPOINT_URL'], aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'], aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto').put_object(Bucket=os.environ['R2_BUCKET_NAME'], Key='${HB_KEY}', Body=sys.stdin.buffer.read())" \
        2>/dev/null || true
    done
  ) &
  echo "[entrypoint] heartbeat+PLOG -> state/heartbeat_${NODE_NAME:-$(hostname)}.txt"
fi

# --- sidecar supervisor (foreground-ish forever loop) ---------------------------
echo "[entrypoint] launching sidecar supervisor"
while true; do
  python "${SIDECAR}"
  RC=$?
  echo "[entrypoint] sidecar exited rc=${RC} - restarting in 10s"
  sleep 10
done
