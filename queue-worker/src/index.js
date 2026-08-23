/**
 * ACE-Stream 5090 BXP — job queue Worker (D1-backed).
 *
 * Pull-model distribution: Salad GPU nodes poll GET /claim; the local
 * dispatcher POSTs batches to /jobs and watches /progress.
 *
 * Auth: every route except GET /health requires header X-Admin-Key === env.ADMIN_KEY.
 * Contract shapes are fixed — sidecar (worker_sidecar.py) and dispatcher
 * (dispatch.py) are coded against exactly these routes.
 *
 * Bindings: env.DB (D1), env.ADMIN_KEY (secret via `wrangler secret put`).
 */

const JSON_HEADERS = { "Content-Type": "application/json" };

function json(body, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: JSON_HEADERS });
}

function err(message, status = 400) {
  return json({ error: message }, status);
}

// ---------------------------------------------------------------- auth
function checkAuth(request, env) {
  const key = request.headers.get("X-Admin-Key");
  if (!env.ADMIN_KEY || !key || key !== env.ADMIN_KEY) {
    return err("unauthorized", 401);
  }
  return null;
}

// ---------------------------------------------------------------- helpers
function now() {
  return Math.floor(Date.now() / 1000);
}

/** Validate one incoming job object; returns [normalizedJob, errorMsg]. */
function normalizeJob(raw) {
  if (!raw || typeof raw !== "object") return [null, "job is not an object"];
  const session = typeof raw.session === "string" ? raw.session.trim().slice(0, 200) : "";
  if (!session) return [null, "missing session"];
  let prompt = typeof raw.prompt === "string" ? raw.prompt.trim() : "";
  if (!prompt) prompt = "Cinematic industrial background layer";
  if (prompt.length > 2000) return [null, "prompt exceeds 2000 chars"];
  let lyrics = typeof raw.lyrics === "string" ? raw.lyrics : "";
  if (lyrics.length > 20000) return [null, "lyrics exceed 20000 chars"];
  let duration = Number.parseInt(raw.duration, 10);
  if (!Number.isFinite(duration)) duration = 240;
  duration = Math.min(600, Math.max(10, duration));
  const filename = typeof raw.output_filename === "string" && raw.output_filename.trim()
    ? raw.output_filename.trim()
    : null;
  if (!filename) return [null, "missing output_filename"];
  const idText = raw.id !== undefined && raw.id !== null ? String(raw.id).slice(0, 64) : null;
  if (!idText) return [null, "missing id"];
  return [{
    id_text: idText,
    session,
    prompt,
    lyrics,
    duration,
    output_filename: filename,
  }, null];
}

// ---------------------------------------------------------------- routes
async function handleJobs(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return err("malformed JSON body", 400);
  }
  const jobs = Array.isArray(body?.jobs) ? body.jobs : null;
  if (!jobs) return err('body must be {"jobs":[...]}', 400);
  if (jobs.length > 1000) return err(`batch too large (${jobs.length} > 1000)`, 413);

  let queued = 0;
  let duplicates = 0;
  for (const raw of jobs) {
    const [job, problem] = normalizeJob(raw);
    if (problem) return err(`bad job: ${problem}`, 422);
    try {
      const res = await env.DB.prepare(
        `INSERT OR IGNORE INTO jobs (id_text, session, prompt, lyrics, duration, output_filename)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)`
      )
        .bind(job.id_text, job.session, job.prompt, job.lyrics, job.duration, job.output_filename)
        .run();
      // D1 meta.changes == 0 means UNIQUE(session,output_filename) hit -> duplicate
      if ((res.meta?.changes ?? 0) > 0) queued += 1;
      else duplicates += 1;
    } catch (e) {
      return err(`d1 insert failed: ${e.message}`, 500);
    }
  }
  return json({ queued, duplicates });
}

async function handleClaim(url, env) {
  const workerId = (url.searchParams.get("worker_id") || "unknown-node").slice(0, 128);
  const t = now();

  // 1) reclaim stale leases (>30 min claimed = node died), max 3 attempts
  try {
    await env.DB.prepare(
      `UPDATE jobs SET status='failed', error='lease expired on final attempt'
       WHERE status='claimed' AND attempts >= 3 AND claimed_at IS NOT NULL
         AND claimed_at < ?1`
    ).bind(t - 1800).run();
    await env.DB.prepare(
      `UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL
       WHERE status='claimed' AND attempts < 3 AND claimed_at IS NOT NULL
         AND claimed_at < ?1`
    ).bind(t - 1800).run();
  } catch (e) {
    return err(`stale-lease reclaim failed: ${e.message}`, 500);
  }

  // 2) atomically claim ONE oldest pending job (single statement)
  try {
    const res = await env.DB.prepare(
      `UPDATE jobs
         SET status='claimed', claimed_by=?1, claimed_at=?2, attempts=attempts+1
       WHERE id = (
         SELECT id FROM jobs WHERE status='pending' ORDER BY id LIMIT 1
       )
       RETURNING id_text AS id, session, prompt, lyrics, duration, output_filename`
    ).bind(workerId, t).all();
    const job = res.results && res.results.length > 0 ? res.results[0] : null;
    return json({ job });
  } catch (e) {
    return err(`claim failed: ${e.message}`, 500);
  }
}

async function handleComplete(request, env) {
  let body;
  try {
    body = await request.json();
  } catch {
    return err("malformed JSON body", 400);
  }
  const jobId = typeof body?.job_id === "string" ? body.job_id : null;
  if (!jobId) return err("missing job_id", 400);

  try {
    if (body.ok === true) {
      const res = await env.DB.prepare(
        `UPDATE jobs SET status='done', r2_key=?2 WHERE id_text=?1 AND status='claimed'`
      ).bind(jobId, body.r2_key || null).run();
      if ((res.meta?.changes ?? 0) === 0) {
        return json({ error: "no active claim" }, 409);
      }
      return json({ requeued: false });
    }

    // failure path: retry up to 3 attempts, else mark failed (single conditional UPDATE)
    await env.DB.prepare(
      `UPDATE jobs SET status=CASE WHEN attempts<3 THEN 'pending' ELSE 'failed' END,
         claimed_by=NULL, claimed_at=NULL, error=?2
       WHERE id_text=?1 AND status='claimed'`
    ).bind(jobId, String(body.error || "unknown").slice(0, 500)).run();
    const row = await env.DB.prepare(
      `SELECT status FROM jobs WHERE id_text=?1`
    ).bind(jobId).first();
    return json({ requeued: row?.status === "pending" });
  } catch (e) {
    return err(`complete failed: ${e.message}`, 500);
  }
}

async function handleProgress(url, env) {
  const session = url.searchParams.get("session");
  if (!session) return err("missing ?session=", 400);
  try {
    const res = await env.DB.prepare(
      `SELECT status, COUNT(*) AS c FROM jobs WHERE session=?1 GROUP BY status`
    ).bind(session).all();
    const counts = { total: 0, done: 0, failed: 0, pending: 0, claimed: 0 };
    for (const row of res.results || []) {
      if (row.status in counts) counts[row.status] = row.c;
      counts.total += row.c;
    }
    return json(counts);
  } catch (e) {
    return err(`progress failed: ${e.message}`, 500);
  }
}

// ---------------------------------------------------------------- router
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname.replace(/\/+$/, "") || "/";

    try {
      if (path === "/health" && request.method === "GET") {
        return json({ ok: true, service: "acestream-queue" });
      }

      const denied = checkAuth(request, env);
      if (denied) return denied;

      switch (true) {
        case path === "/jobs" && request.method === "POST":
          return await handleJobs(request, env);
        case path === "/claim" && request.method === "GET":
          return await handleClaim(url, env);
        case path === "/complete" && request.method === "POST":
          return await handleComplete(request, env);
        case path === "/progress" && request.method === "GET":
          return await handleProgress(url, env);
        default:
          return err(`no route: ${request.method} ${path}`, 404);
      }
    } catch (e) {
      return err(`internal: ${e.message}`, 500);
    }
  },
};
