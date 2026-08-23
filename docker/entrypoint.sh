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

# --- sidecar supervisor (foreground-ish forever loop) ---------------------------
echo "[entrypoint] launching sidecar supervisor"
while true; do
  python "${SIDECAR}"
  RC=$?
  echo "[entrypoint] sidecar exited rc=${RC} - restarting in 10s"
  sleep 10
done
