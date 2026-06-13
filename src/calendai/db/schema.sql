-- CalendAI SQLite schema. Single source of truth; applied idempotently at startup.
-- All timestamps are ISO-8601 UTC strings.

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    email       TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL DEFAULT '',
    timezone    TEXT NOT NULL DEFAULT 'Asia/Kolkata',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS oauth_tokens (
    user_id     TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    provider    TEXT NOT NULL DEFAULT 'google',
    token_blob  BLOB NOT NULL,  -- Fernet-encrypted JSON {access_token, refresh_token, expiry}
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token TEXT,
    role          TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content       TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    fact_type     TEXT NOT NULL CHECK (fact_type IN ('rule', 'contact', 'preference')),
    key           TEXT NOT NULL,
    value         TEXT NOT NULL,  -- JSON payload read by enforcement code
    statement     TEXT NOT NULL,  -- human-readable line injected into the system prompt
    provenance    TEXT NOT NULL,  -- what the user said that created this fact
    active        INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER REFERENCES memory_facts(id),
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

-- At most one ACTIVE fact per (user, key); supersession deactivates the old row.
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_active_key
    ON memory_facts(user_id, key) WHERE active = 1;

-- Destructive-action consent that must survive between HTTP requests. The web
-- app rebuilds the agent loop (and its in-memory ConfirmationGate) on every
-- /api/chat call, so a token issued on the "delete?" turn would otherwise be
-- lost before the "yes" turn. A row here is single-shot: it is consumed (and
-- deleted) at the start of the user's very next turn.
CREATE TABLE IF NOT EXISTS pending_confirmations (
    token       TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    action      TEXT NOT NULL CHECK (action IN ('update_event', 'delete_event')),
    fingerprint TEXT NOT NULL,  -- sha256 of the canonical args; consent binds to these exact args
    summary     TEXT NOT NULL,  -- canonical args JSON, re-shown to the model next turn
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_conf_user ON pending_confirmations(user_id);

CREATE TABLE IF NOT EXISTS trace_requests (
    request_id   TEXT PRIMARY KEY,
    user_id      TEXT,
    user_message TEXT,
    started_at   TEXT NOT NULL,
    ended_at     TEXT
);

CREATE TABLE IF NOT EXISTS trace_spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id  TEXT NOT NULL REFERENCES trace_requests(request_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL,  -- llm_call | tool_call | memory_op | decision
    name        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    payload     TEXT NOT NULL DEFAULT '{}',  -- JSON
    rationale   TEXT
);

CREATE INDEX IF NOT EXISTS idx_spans_request ON trace_spans(request_id);
