-- =============================================================================
-- Illustrative schema for an AI-companion platform.
-- Target: PostgreSQL 15+ with pgvector and pg_trgm available.
-- This file is self-contained and can be executed against a fresh database.
-- It is a portfolio illustration, not a copy of any production system.
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;       -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pg_trgm;        -- trigram similarity for fuzzy matching
CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector: embeddings + ANN indexes

-- -----------------------------------------------------------------------------
-- Reusable timestamp trigger.
-- One function, attached to every table that needs an updated_at column.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION trigger_set_timestamp()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

-- -----------------------------------------------------------------------------
-- App-level "current user" accessor.
--
-- For portability we read the user id from a GUC set by the connection pooler
-- or middleware, e.g. SET LOCAL app.user_id = '...'. Real systems frequently
-- delegate this to a managed auth layer (a JWT claim resolver). The pattern
-- shown here is intentionally backend-agnostic.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION current_app_user_id()
RETURNS uuid
LANGUAGE sql
STABLE
AS $$
    SELECT NULLIF(current_setting('app.user_id', true), '')::uuid;
$$;


-- =============================================================================
-- USERS
-- Owner of all user-scoped data. One row per authenticated principal.
-- =============================================================================
CREATE TABLE users (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    email           citext      UNIQUE,                    -- citext keeps us honest about case
    display_name    text        NOT NULL,
    preferences     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE users IS 'Authenticated end users. PK is generated server-side.';

-- citext requires the extension; declared here to keep the example runnable.
CREATE EXTENSION IF NOT EXISTS citext;
ALTER TABLE users
    ALTER COLUMN email TYPE citext USING email::citext;

CREATE TRIGGER trg_users_set_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();


-- =============================================================================
-- COMPANIONS
-- Catalog of AI personas. Read-mostly; written by content tooling.
-- personality_flavor is intentionally jsonb because the shape evolves:
-- adding a new behavioral axis must not require an online migration.
-- =============================================================================
CREATE TABLE companions (
    id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    slug                text        NOT NULL UNIQUE,
    display_name        text        NOT NULL,
    personality_flavor  jsonb       NOT NULL DEFAULT '{}'::jsonb,
    is_active           boolean     NOT NULL DEFAULT true,
    created_at          timestamptz NOT NULL DEFAULT now(),
    updated_at          timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE companions IS 'AI persona catalog. Hot-read, cold-write.';

-- jsonb_path_ops is smaller and faster than the default jsonb_ops when we only
-- need containment queries: WHERE personality_flavor @> '{"voice":"warm"}'.
CREATE INDEX idx_companions_flavor_path
    ON companions USING GIN (personality_flavor jsonb_path_ops);

-- Most listing screens want active personas only; partial index keeps it tight.
CREATE INDEX idx_companions_active
    ON companions (created_at DESC)
    WHERE is_active = true;

CREATE TRIGGER trg_companions_set_updated_at
    BEFORE UPDATE ON companions
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();


-- =============================================================================
-- USER_MATCHES
-- The "chat list row." Composite PK (user_id, companion_id).
--
-- DENORMALIZED HOT FIELDS: last_message, last_message_at, unread_count.
-- These are derived from the messages table but materialized here to make the
-- chat-list endpoint a single-index scan. Maintained by trg_update_match_on_message.
-- =============================================================================
CREATE TABLE user_matches (
    user_id          uuid        NOT NULL REFERENCES users(id)      ON DELETE CASCADE,
    companion_id     uuid        NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    is_active        boolean     NOT NULL DEFAULT true,

    -- Denormalized projections of the latest message for this pair.
    last_message     text,                                          -- truncated to ~100 chars at write time
    last_message_at  timestamptz,
    unread_count     integer     NOT NULL DEFAULT 0 CHECK (unread_count >= 0),

    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now(),

    PRIMARY KEY (user_id, companion_id)
);
COMMENT ON TABLE user_matches IS
    'Per-(user, companion) chat header with denormalized last-message state.';

-- Powers the chat list: WHERE user_id = ? AND is_active ORDER BY last_message_at DESC.
-- Partial index avoids bloat from soft-deleted matches.
CREATE INDEX idx_matches_user_sorted
    ON user_matches (user_id, last_message_at DESC NULLS LAST)
    WHERE is_active = true;

CREATE TRIGGER trg_matches_set_updated_at
    BEFORE UPDATE ON user_matches
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();


-- =============================================================================
-- DIALOG_SESSIONS
-- Bounds a "conversation episode." Closed sessions get a summary + memories.
-- One active session per (user, companion); enforced via partial unique index.
-- =============================================================================
CREATE TABLE dialog_sessions (
    id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         uuid        NOT NULL REFERENCES users(id)      ON DELETE CASCADE,
    companion_id    uuid        NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    is_active       boolean     NOT NULL DEFAULT true,
    recap           text,                                          -- LLM-generated summary, set on close
    started_at      timestamptz NOT NULL DEFAULT now(),
    closed_at       timestamptz
);
COMMENT ON TABLE dialog_sessions IS
    'Conversation episode. At most one active per (user, companion).';

-- Enforce "one active session per pair" without locking the whole table.
CREATE UNIQUE INDEX uniq_active_session_per_pair
    ON dialog_sessions (user_id, companion_id)
    WHERE is_active = true;

-- Most lookups are: "give me the active session for this pair."
CREATE INDEX idx_dialog_sessions_active
    ON dialog_sessions (user_id, companion_id)
    WHERE is_active = true;


-- =============================================================================
-- MESSAGES
-- Append-only conversation log. The dominant read pattern is:
--   WHERE user_id = ? AND companion_id = ? ORDER BY created_at DESC LIMIT N.
--
-- A generated tsvector column gives us deterministic FTS without a trigger.
-- Use 'simple' so we don't bake a single language into multilingual content.
-- =============================================================================
CREATE TABLE messages (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid        NOT NULL REFERENCES users(id)            ON DELETE CASCADE,
    companion_id  uuid        NOT NULL REFERENCES companions(id)       ON DELETE CASCADE,
    session_id    uuid        REFERENCES dialog_sessions(id)           ON DELETE SET NULL,
    role          text        NOT NULL CHECK (role IN ('user', 'companion', 'system')),
    content       text        NOT NULL,
    content_fts   tsvector    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    is_read       boolean     NOT NULL DEFAULT false,
    created_at    timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE messages IS 'Append-only conversation log.';

-- The hot index. Composite ordering matches ORDER BY created_at DESC.
CREATE INDEX idx_messages_user_companion_time
    ON messages (user_id, companion_id, created_at DESC);

-- FTS index. GIN over a generated tsvector is the textbook pattern.
CREATE INDEX idx_messages_fts
    ON messages USING GIN (content_fts);

-- Trigram index supports cheap "did you mean / contains" lookups in admin UIs
-- without paying the full FTS cost.
CREATE INDEX idx_messages_content_trgm
    ON messages USING GIN (content gin_trgm_ops);


-- =============================================================================
-- DENORMALIZATION TRIGGER
-- Maintains the chat-list projection on user_matches whenever a message is
-- written. Idempotent: it always reflects the current row's content; later
-- inserts overwrite earlier denormalized state, which is what we want.
--
-- Race notes:
--   * Two concurrent inserts: the trigger runs in each transaction's snapshot.
--     The "winner" is whichever commits last; ordering follows commit order,
--     not statement order. This matches user expectations for chat UIs.
--   * unread_count is incremented only for messages authored by the companion;
--     decrement happens in mark_chat_read() (see below).
-- =============================================================================
CREATE OR REPLACE FUNCTION fn_update_match_on_message()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
DECLARE
    truncated_message text := left(NEW.content, 100);
BEGIN
    INSERT INTO user_matches (user_id, companion_id, last_message, last_message_at, unread_count)
    VALUES (
        NEW.user_id,
        NEW.companion_id,
        truncated_message,
        NEW.created_at,
        CASE WHEN NEW.role = 'companion' THEN 1 ELSE 0 END
    )
    ON CONFLICT (user_id, companion_id) DO UPDATE
        SET last_message    = EXCLUDED.last_message,
            last_message_at = EXCLUDED.last_message_at,
            unread_count    = user_matches.unread_count
                              + CASE WHEN NEW.role = 'companion' THEN 1 ELSE 0 END,
            updated_at      = now()
        WHERE user_matches.last_message_at IS NULL
           OR user_matches.last_message_at <= NEW.created_at;
    RETURN NEW;
END;
$$;

CREATE TRIGGER trg_update_match_on_message
    AFTER INSERT ON messages
    FOR EACH ROW EXECUTE FUNCTION fn_update_match_on_message();


-- =============================================================================
-- MEMORIES
-- Long-term, per-(user, companion) facts extracted from closed dialog sessions.
-- Each row carries:
--   * content         - the canonical fact statement
--   * context_excerpt - the surrounding utterance for retrieval grounding
--   * embedding       - dense vector (Qwen3-style 1024d, change to your model)
--   * content_fts     - keyword fallback / hybrid retrieval
-- =============================================================================
CREATE TABLE memories (
    id                  uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid        NOT NULL REFERENCES users(id)            ON DELETE CASCADE,
    companion_id        uuid        NOT NULL REFERENCES companions(id)       ON DELETE CASCADE,
    source_session_id   uuid        REFERENCES dialog_sessions(id)           ON DELETE SET NULL,
    content             text        NOT NULL,
    context_excerpt     text,
    embedding           vector(1024),
    content_fts         tsvector    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    access_count        integer     NOT NULL DEFAULT 0,
    last_accessed_at    timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE memories IS 'Long-term episodic / semantic memory. Source of truth.';

-- Composite filter index: every retrieval is per-(user, companion).
-- A leading user_id is also what makes our RLS predicate cheap.
CREATE INDEX idx_memories_user_companion
    ON memories (user_id, companion_id, created_at DESC);

-- HNSW index for ANN. Cosine distance matches normalized embedding models.
-- Tuning notes:
--   m              = 16  : graph degree. Larger -> better recall, more memory.
--   ef_construction = 64 : build-time search width. Larger -> better recall,
--                          slower build. Doubling is cheap up to ~200.
-- Read-time recall is governed by SET hnsw.ef_search = N; in the session.
CREATE INDEX idx_memories_embedding_hnsw
    ON memories USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- FTS lane of the hybrid retriever.
CREATE INDEX idx_memories_fts
    ON memories USING GIN (content_fts);


-- =============================================================================
-- SCHEDULES
-- Per-companion weekly availability. Used by the proactive engagement system
-- to decide when a persona should initiate or respond. Many rows per companion;
-- read pattern is "show me today's slots", so day_of_week leads.
-- =============================================================================
CREATE TABLE schedules (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    companion_id  uuid        NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    day_of_week   smallint    NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),
    starts_at     time        NOT NULL,
    ends_at       time        NOT NULL,
    activity      text        NOT NULL,
    CHECK (ends_at > starts_at)
);
COMMENT ON TABLE schedules IS 'Weekly availability slots per companion.';

CREATE INDEX idx_schedules_companion_day
    ON schedules (companion_id, day_of_week, starts_at);


-- =============================================================================
-- STORIES
-- Ephemeral, pre-generated short videos. status drives a state machine:
--   pending -> generating -> ready -> expired | failed.
-- We only ever query "ready and not yet expired" for a given companion.
-- =============================================================================
CREATE TABLE stories (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    companion_id  uuid        NOT NULL REFERENCES companions(id) ON DELETE CASCADE,
    status        text        NOT NULL CHECK (status IN ('pending','generating','ready','expired','failed')),
    video_url     text,
    caption       text,
    expires_at    timestamptz,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE stories IS 'Short-form generated media. State-machine driven.';

-- Dominant read: list ready, non-expired stories for a companion.
CREATE INDEX idx_stories_ready_by_companion
    ON stories (companion_id, created_at DESC)
    WHERE status = 'ready';

-- Background reaper: anything past its TTL.
CREATE INDEX idx_stories_expiring
    ON stories (expires_at)
    WHERE status = 'ready';

CREATE TRIGGER trg_stories_set_updated_at
    BEFORE UPDATE ON stories
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();


-- =============================================================================
-- STORY_VIEWS
-- (story_id, user_id) is unique - we only count one view per user per story.
-- =============================================================================
CREATE TABLE story_views (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id    uuid        NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    user_id     uuid        NOT NULL REFERENCES users(id)   ON DELETE CASCADE,
    liked       boolean     NOT NULL DEFAULT false,
    viewed_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (story_id, user_id)
);
CREATE INDEX idx_story_views_user ON story_views (user_id, viewed_at DESC);


-- =============================================================================
-- GIFTS (catalog + ledger)
-- Catalog is small and read-mostly. Purchases are append-only; we never UPDATE.
-- =============================================================================
CREATE TABLE gift_catalog (
    id           uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    sku          text        NOT NULL UNIQUE,
    display_name text        NOT NULL,
    price_cents  integer     NOT NULL CHECK (price_cents > 0),
    is_active    boolean     NOT NULL DEFAULT true
);

CREATE TABLE gift_purchases (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       uuid        NOT NULL REFERENCES users(id)         ON DELETE RESTRICT,
    companion_id  uuid        NOT NULL REFERENCES companions(id)    ON DELETE RESTRICT,
    gift_id       uuid        NOT NULL REFERENCES gift_catalog(id)  ON DELETE RESTRICT,
    amount_cents  integer     NOT NULL CHECK (amount_cents >= 0),
    purchased_at  timestamptz NOT NULL DEFAULT now()
);
COMMENT ON TABLE gift_purchases IS 'Append-only ledger. Use restrict on FKs to keep history.';

CREATE INDEX idx_gift_purchases_user_time
    ON gift_purchases (user_id, purchased_at DESC);


-- =============================================================================
-- VIOLATIONS
-- Small, append-mostly moderation log.
-- =============================================================================
CREATE TABLE violations (
    id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     uuid        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category    text        NOT NULL,
    details     text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX idx_violations_user_time ON violations (user_id, created_at DESC);


-- =============================================================================
-- BACKGROUND-JOB QUEUE (illustrative)
-- Pattern: SELECT ... FOR UPDATE SKIP LOCKED to claim work atomically across
-- N concurrent workers without an external broker.
-- =============================================================================
CREATE TABLE story_generation_jobs (
    id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
    story_id      uuid        NOT NULL REFERENCES stories(id) ON DELETE CASCADE,
    status        text        NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending','running','done','failed')),
    locked_until  timestamptz,
    attempts      smallint    NOT NULL DEFAULT 0,
    last_error    text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- Partial index over only the queue head -> tiny, cache-friendly.
CREATE INDEX idx_story_jobs_pending
    ON story_generation_jobs (created_at)
    WHERE status = 'pending';

CREATE TRIGGER trg_story_jobs_set_updated_at
    BEFORE UPDATE ON story_generation_jobs
    FOR EACH ROW EXECUTE FUNCTION trigger_set_timestamp();


-- =============================================================================
-- ROW LEVEL SECURITY
-- Pattern shown: every user-scoped table is RLS-enforced. The predicate is
-- always a direct equality on user_id, which lets the planner combine it with
-- the leading-column composite index.
--
-- A "service role" connection that needs to bypass RLS would use a role with
-- BYPASSRLS or run with security definer functions.
-- =============================================================================
ALTER TABLE users           ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_matches    ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages        ENABLE ROW LEVEL SECURITY;
ALTER TABLE dialog_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE memories        ENABLE ROW LEVEL SECURITY;
ALTER TABLE story_views     ENABLE ROW LEVEL SECURITY;
ALTER TABLE gift_purchases  ENABLE ROW LEVEL SECURITY;
ALTER TABLE violations      ENABLE ROW LEVEL SECURITY;

-- Self-row policy on users.
CREATE POLICY users_self_access ON users
    FOR ALL
    USING      (id = current_app_user_id())
    WITH CHECK (id = current_app_user_id());

-- One canonical pattern reused across user-scoped tables.
CREATE POLICY user_matches_self    ON user_matches
    FOR ALL USING (user_id = current_app_user_id())
            WITH CHECK (user_id = current_app_user_id());

CREATE POLICY messages_self        ON messages
    FOR ALL USING (user_id = current_app_user_id())
            WITH CHECK (user_id = current_app_user_id());

CREATE POLICY dialog_sessions_self ON dialog_sessions
    FOR ALL USING (user_id = current_app_user_id())
            WITH CHECK (user_id = current_app_user_id());

CREATE POLICY memories_self        ON memories
    FOR ALL USING (user_id = current_app_user_id())
            WITH CHECK (user_id = current_app_user_id());

CREATE POLICY story_views_self     ON story_views
    FOR ALL USING (user_id = current_app_user_id())
            WITH CHECK (user_id = current_app_user_id());

CREATE POLICY gift_purchases_self  ON gift_purchases
    FOR SELECT USING (user_id = current_app_user_id());

CREATE POLICY violations_self      ON violations
    FOR SELECT USING (user_id = current_app_user_id());


-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- mark_chat_read: idempotent, atomic reset of unread counter on a match.
-- Returns the row that was updated, NULL if no such match exists.
CREATE OR REPLACE FUNCTION mark_chat_read(p_user_id uuid, p_companion_id uuid)
RETURNS user_matches
LANGUAGE sql
AS $$
    UPDATE user_matches
       SET unread_count = 0,
           updated_at   = now()
     WHERE user_id      = p_user_id
       AND companion_id = p_companion_id
    RETURNING *;
$$;

-- claim_pending_jobs: dequeue up to N jobs with SKIP LOCKED.
-- Multiple workers can call this concurrently; each claims a disjoint subset.
CREATE OR REPLACE FUNCTION claim_pending_jobs(p_limit integer)
RETURNS SETOF story_generation_jobs
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    UPDATE story_generation_jobs sj
       SET status       = 'running',
           locked_until = now() + interval '5 minutes',
           attempts     = sj.attempts + 1,
           updated_at   = now()
     WHERE sj.id IN (
        SELECT id FROM story_generation_jobs
         WHERE status = 'pending'
         ORDER BY created_at
         LIMIT p_limit
         FOR UPDATE SKIP LOCKED
     )
    RETURNING *;
END;
$$;
