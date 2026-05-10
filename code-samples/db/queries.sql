-- =============================================================================
-- Representative non-trivial queries against the illustrative schema.
-- Each block includes the access pattern, the query, and an EXPLAIN sketch
-- showing the plan a senior engineer should expect after the schema is loaded
-- and ANALYZEd.
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Chat list for a user.
--
-- Naive implementation: SELECT MAX(created_at) FROM messages GROUP BY ...
-- That degrades to O(messages) per user. We materialize last_message_at on
-- user_matches via trg_update_match_on_message and serve the page from a
-- single-index scan.
-- -----------------------------------------------------------------------------
SELECT m.companion_id,
       c.display_name,
       m.last_message,
       m.last_message_at,
       m.unread_count
  FROM user_matches m
  JOIN companions   c ON c.id = m.companion_id
 WHERE m.user_id    = $1
   AND m.is_active  = true
 ORDER BY m.last_message_at DESC NULLS LAST
 LIMIT 50;
-- Expected plan:
--   Limit
--     -> Nested Loop
--          -> Index Scan using idx_matches_user_sorted on user_matches
--                Index Cond: (user_id = $1)
--          -> Index Scan using companions_pkey on companions
--                Index Cond: (id = m.companion_id)
-- No sort node. No Seq Scan. Reads bounded by LIMIT.


-- -----------------------------------------------------------------------------
-- 2. Recent message history for a single conversation.
--
-- The composite index (user_id, companion_id, created_at DESC) lets us serve
-- the chat scroll in reverse-chronological order without a Sort node.
-- -----------------------------------------------------------------------------
SELECT id, role, content, created_at
  FROM messages
 WHERE user_id      = $1
   AND companion_id = $2
 ORDER BY created_at DESC
 LIMIT 50;
-- Expected plan:
--   Limit
--     -> Index Scan Backward using idx_messages_user_companion_time on messages
--          Index Cond: ((user_id = $1) AND (companion_id = $2))


-- -----------------------------------------------------------------------------
-- 3. Memory retrieval: HNSW kNN with a per-user filter.
--
-- The filter columns are NOT in the HNSW index; pgvector applies them after
-- approximate search. With small-to-medium memory tables, this is fine.
-- For very large tables, partition by user_id or maintain per-companion
-- HNSW indexes.
--
-- ef_search controls recall vs latency at query time.
-- -----------------------------------------------------------------------------
SET LOCAL hnsw.ef_search = 80;

SELECT id,
       content,
       context_excerpt,
       1 - (embedding <=> $1::vector) AS similarity
  FROM memories
 WHERE user_id      = $2
   AND companion_id = $3
 ORDER BY embedding <=> $1::vector
 LIMIT 10;
-- Expected plan:
--   Limit
--     -> Index Scan using idx_memories_embedding_hnsw on memories
--          Order By: (embedding <=> $1)
--          Filter:   ((user_id = $2) AND (companion_id = $3))


-- -----------------------------------------------------------------------------
-- 4. Hybrid search: pgvector kNN UNION-ed with FTS, fused via RRF.
--
-- Reciprocal Rank Fusion combines two ranked lists without needing comparable
-- score scales. k is a smoothing constant; 60 is the canonical default.
--
-- Each leg returns at most TOP_K rows, so the working set is small and the
-- final ORDER BY runs against ~2*TOP_K rows. This is the right shape for an
-- LLM retrieval step.
-- -----------------------------------------------------------------------------
WITH params AS (
    SELECT $1::vector AS query_vec,
           $2::text   AS query_text,
           $3::uuid   AS user_id,
           $4::uuid   AS companion_id,
           20::int    AS top_k,
           60::int    AS rrf_k
),
vec AS (
    SELECT m.id,
           ROW_NUMBER() OVER (ORDER BY m.embedding <=> p.query_vec) AS rnk
      FROM memories m, params p
     WHERE m.user_id      = p.user_id
       AND m.companion_id = p.companion_id
     ORDER BY m.embedding <=> p.query_vec
     LIMIT (SELECT top_k FROM params)
),
fts AS (
    SELECT m.id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(m.content_fts, plainto_tsquery('simple', p.query_text)) DESC
           ) AS rnk
      FROM memories m, params p
     WHERE m.user_id      = p.user_id
       AND m.companion_id = p.companion_id
       AND m.content_fts @@ plainto_tsquery('simple', p.query_text)
     LIMIT (SELECT top_k FROM params)
),
fused AS (
    SELECT id, SUM(score) AS score
      FROM (
           SELECT v.id,
                  1.0 / ((SELECT rrf_k FROM params) + v.rnk) AS score
             FROM vec v
            UNION ALL
           SELECT f.id,
                  1.0 / ((SELECT rrf_k FROM params) + f.rnk) AS score
             FROM fts f
      ) t
     GROUP BY id
)
SELECT m.id, m.content, m.context_excerpt, fused.score
  FROM fused
  JOIN memories m ON m.id = fused.id
 ORDER BY fused.score DESC
 LIMIT 10;
-- Expected plan:
--   Sort (fused.score DESC) over a small CTE result.
--   Each leg uses its own index:
--     vec -> Index Scan using idx_memories_embedding_hnsw
--     fts -> Bitmap Heap Scan on memories  /  Bitmap Index Scan on idx_memories_fts
--   Final join: Hash Join or Nested Loop on memories_pkey.


-- -----------------------------------------------------------------------------
-- 5. Mark a chat as read - idempotent, atomic, RLS-friendly.
-- -----------------------------------------------------------------------------
SELECT * FROM mark_chat_read($1, $2);
-- Expected plan:
--   Update on user_matches
--     -> Index Scan using user_matches_pkey on user_matches
--          Index Cond: ((user_id = $1) AND (companion_id = $2))


-- -----------------------------------------------------------------------------
-- 6. Atomic job claim with FOR UPDATE SKIP LOCKED.
--
-- N parallel workers can call this concurrently. Each gets a disjoint slice.
-- No row-level lock contention; no global mutex.
-- -----------------------------------------------------------------------------
SELECT * FROM claim_pending_jobs(10);
-- Or, inline:
WITH claimed AS (
    SELECT id
      FROM story_generation_jobs
     WHERE status = 'pending'
     ORDER BY created_at
     LIMIT 10
     FOR UPDATE SKIP LOCKED
)
UPDATE story_generation_jobs sj
   SET status       = 'running',
       locked_until = now() + interval '5 minutes',
       attempts     = sj.attempts + 1,
       updated_at   = now()
  FROM claimed
 WHERE sj.id = claimed.id
RETURNING sj.*;
-- Expected plan:
--   Update -> Hash Join
--               -> CTE Scan on claimed
--                    -> Limit
--                         -> LockRows
--                              -> Index Scan using idx_story_jobs_pending


-- -----------------------------------------------------------------------------
-- 7. Stories feed: ready, non-expired, ordered.
--
-- Uses the partial index idx_stories_ready_by_companion - we never even
-- consider rows in pending / generating / failed states.
-- -----------------------------------------------------------------------------
SELECT id, video_url, caption, expires_at
  FROM stories
 WHERE companion_id = $1
   AND status       = 'ready'
   AND (expires_at IS NULL OR expires_at > now())
 ORDER BY created_at DESC
 LIMIT 20;
-- Expected plan:
--   Limit
--     -> Index Scan using idx_stories_ready_by_companion on stories
--          Index Cond: (companion_id = $1)
--          Filter: ((expires_at IS NULL) OR (expires_at > now()))


-- -----------------------------------------------------------------------------
-- 8. JSONB containment query on companions.
--
-- We index personality_flavor with jsonb_path_ops; @> is the matched op class.
-- -----------------------------------------------------------------------------
SELECT id, slug, display_name
  FROM companions
 WHERE personality_flavor @> '{"voice":"warm"}'
   AND is_active = true;
-- Expected plan:
--   Bitmap Heap Scan on companions
--     Recheck Cond: (personality_flavor @> '{"voice":"warm"}')
--     Filter: is_active
--     -> Bitmap Index Scan on idx_companions_flavor_path
