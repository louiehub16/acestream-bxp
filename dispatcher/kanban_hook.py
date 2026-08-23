#!/usr/bin/env python3
"""
ACE-Stream 5090 BXP - optional kanban progress hook.

When KANBAN_PROGRESS_URL is set, `dispatch.py watch` POSTs a progress snapshot
to that URL on every poll tick via post_progress(session, stats). The payload
mirrors the kanban worker contract discovered in PCBGenius_Local
(supervisor/supervisor.py -> post_kanban):

    endpoint  e.g. https://kanban.<project>.workers.dev/api/update
    body      {"agent", "status", "feature", "message"[, "phase"]}
              (+ our generic {"service", "session", "stats", "ts"} extras)
    auth      none - instead the board's Cloudflare WAF REJECTS non-browser
              User-Agents with 403 / error 1010, so a Chrome UA header is
              mandatory (same fix as the Salad API caller).

Contract: silent best-effort. A dead board must never kill or spam a watch
loop - no exceptions escape, nothing is printed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error  # noqa: F401  (kept so callers can introspect failure modes)
import urllib.request

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 "
              "Safari/537.36")
DEFAULT_AGENT = "acestream-dispatcher"
TIMEOUT_SECONDS = 10


def progress_url() -> str:
    """KANBAN_PROGRESS_URL from the environment ('' = hook disabled)."""
    return (os.environ.get("KANBAN_PROGRESS_URL")
            or os.environ.get("KANBAN_URL", "")).strip()


def post_progress(session: str, stats: dict) -> bool:
    """POST one watch-tick progress snapshot to the kanban board.

    Returns True on a 2xx response, False for any other outcome (URL unset,
    network error, non-2xx). Never raises, never prints.
    """
    url = progress_url()
    if not url:
        return False

    done = stats.get("done", 0)
    failed = stats.get("failed", 0)
    total = stats.get("total", 0)
    body = {
        # generic, self-describing payload
        "service": "acestream",
        "session": session,
        "stats": stats,
        "ts": time.time(),
        # PCBGenius kanban worker shape (/api/update)
        "agent": os.environ.get("AGENT_NAME", DEFAULT_AGENT),
        "status": "ok",
        "feature": session,
        "message": f"acestream {session}: {done}/{total} done, {failed} failed",
    }
    if total and done + failed >= total:
        body["phase"] = "complete"

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json",
                 "Accept": "application/json",
                 "User-Agent": BROWSER_UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            return 200 <= resp.status < 300
    except Exception:  # noqa: BLE001 - silent best-effort by contract
        return False


if __name__ == "__main__":  # manual smoke test: prints True/False, nothing else
    import sys

    ok = post_progress(sys.argv[1] if len(sys.argv) > 1 else "smoke-test",
                       {"done": 1, "failed": 0, "pending": 0, "claimed": 0,
                        "total": 1})
    print(ok)
