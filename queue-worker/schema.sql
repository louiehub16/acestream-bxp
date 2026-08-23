-- ACE-Stream 5090 BXP job queue - D1 schema
-- Apply with:  npx wrangler d1 execute acestream-queue --remote --file=schema.sql
--
-- Portability note: uses strftime('%s','now') instead of unixepoch() so the
-- schema also executes on older bundled SQLite (<3.38).

CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    id_text         TEXT,                              -- dispatcher's job id string ("0001")
    session         TEXT NOT NULL,                     -- generation_YYYY-MM-DD_HH-MM-SS
    prompt          TEXT NOT NULL,
    lyrics          TEXT DEFAULT '',
    duration        INTEGER DEFAULT 240,
    output_filename TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending|claimed|done|failed
    claimed_by      TEXT,                              -- node hostname
    claimed_at      INTEGER,                           -- epoch seconds
    attempts        INTEGER DEFAULT 0,
    r2_key          TEXT,                              -- <session>/<file>.wav
    error           TEXT,
    created_at      INTEGER DEFAULT (strftime('%s','now')),
    UNIQUE(session, output_filename)
);

CREATE INDEX IF NOT EXISTS idx_jobs_session_status ON jobs(session, status);
CREATE INDEX IF NOT EXISTS idx_jobs_status_id ON jobs(status, id);
