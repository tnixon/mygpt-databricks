-- MyGPT ingest — section-aware chunk, then MERGE into the AI Search source table.
-- Runs after 02_parse_chunk_index.sql (parse -> _doc_elements_stage) and etl/parse_fallback.py
-- (Phase 6: PyMuPDF elements for garbled / script-less documents). Steps 4-9 of the
-- pre-Phase-6 02, unchanged apart from the doc_parse_quality cleanup. Catalog / schema: the
-- job's `catalog` / `schema` parameters.

-- Job parameters (databricks.yml sql_task.parameters): catalog, schema, docs_volume.
USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- 4) Apply element changes: drop elements of re-parsed and out-of-scope (removed file /
--    dropped corpus) documents, then add the freshly parsed ones.
DELETE FROM doc_elements
WHERE doc_id IN (SELECT doc_id FROM _docs_to_parse)
   OR doc_id NOT IN (SELECT doc_id FROM _docs_in_scope);

INSERT INTO doc_elements
  (doc_id, path, corpus, idx, elem_type, page, content, header_text, section)
SELECT doc_id, path, corpus, idx, elem_type, page, content, header_text, section
FROM _doc_elements_stage;

-- Record the parsed versions (n_elements = 0 still counts as parsed, so a document that
-- yields no elements isn't re-parsed every run) and forget out-of-scope documents.
MERGE INTO doc_parse_log t
USING (
  SELECT p.doc_id, p.path, p.corpus, p.size_bytes, p.modification_time,
         current_timestamp() AS parsed_at, count(e.idx) AS n_elements
  FROM _docs_to_parse p
  LEFT JOIN _doc_elements_stage e ON e.doc_id = p.doc_id
  GROUP BY p.doc_id, p.path, p.corpus, p.size_bytes, p.modification_time
) s
ON t.doc_id = s.doc_id
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

DELETE FROM doc_parse_log
WHERE doc_id NOT IN (SELECT doc_id FROM _docs_in_scope);

DELETE FROM doc_parse_quality
WHERE doc_id NOT IN (SELECT doc_id FROM _docs_in_scope);

-- 5) Re-derive ALL chunks from doc_elements (cheap SQL, and deterministic: unchanged
--    documents keep identical elements, so they produce identical chunks).
--    ~1000-char, section-bounded chunks (no overlap).
CREATE OR REPLACE TABLE _doc_chunks_stage AS
WITH body AS (
  -- keep substantive body text; drop headers themselves and tiny noise
  SELECT
    doc_id, path, corpus, idx, page, section, content,
    regexp_extract(path, '([^/]+)$', 1) AS source_name
  FROM doc_elements
  WHERE length(content) >= 80
    AND lower(elem_type) NOT IN ('page_header', 'page_footer', 'page_number')
),
bucketed AS (
  SELECT
    doc_id, path, corpus, source_name, idx, page, section, content,
    -- running char length within (doc, section) -> fixed ~1000-char buckets
    cast(floor(
      (sum(length(content) + 1)
         OVER (PARTITION BY doc_id, section ORDER BY idx
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) - 1)
      / 1000.0) AS INT) AS bucket
  FROM body
),
chunks AS (
  SELECT
    doc_id,
    any_value(path)                                   AS path,
    any_value(source_name)                            AS source_name,
    any_value(corpus)                                 AS corpus,
    section,
    bucket,
    min(idx)                                          AS first_idx,
    min(page)                                         AS page,
    concat_ws(' ', array_sort(collect_list(struct(idx, content)))
                     .content)                        AS chunk_text
  FROM bucketed
  GROUP BY doc_id, section, bucket
)
SELECT
  -- Stable key: path hash (01) + section hash + bucket. 64-bit section hash so two
  -- sections of one document can't collide (a duplicate key would fail the MERGE).
  concat(doc_id, '-', substr(md5(section), 1, 16), '-', bucket) AS chunk_id,
  doc_id,
  path,
  source_name,
  corpus,
  section,
  page,
  'section_chunk'                                                AS elem_type,
  chunk_text
FROM chunks
WHERE length(chunk_text) >= 80;

-- 6) MERGE into the index source: only real changes touch the change feed.
MERGE INTO doc_chunks t
USING _doc_chunks_stage s
ON t.chunk_id = s.chunk_id
WHEN MATCHED AND NOT (
      t.chunk_text  <=> s.chunk_text
  AND t.section     <=> s.section
  AND t.page        <=> s.page
  AND t.source_name <=> s.source_name
  AND t.corpus      <=> s.corpus
  AND t.path        <=> s.path
) THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
WHEN NOT MATCHED BY SOURCE THEN DELETE;

-- 7) Per-dataset section catalogue (port of MyGPT PaperSections) — powers the
--    document/section focus selector and the agent's section validation.
--    'mygpt' (the app's default, unfiltered view) spans every corpus; each corpus
--    also gets its own rows so per-corpus focus selectors can list just its sections.
CREATE OR REPLACE TABLE paper_sections AS
SELECT 'mygpt' AS dataset_name, section AS section_title, count(*) AS section_count
FROM doc_chunks
GROUP BY section
UNION ALL
SELECT corpus AS dataset_name, section AS section_title, count(*) AS section_count
FROM doc_chunks
GROUP BY corpus, section;

-- 8) Drop per-run staging.
DROP TABLE IF EXISTS _docs_in_scope;
DROP TABLE IF EXISTS _docs_to_parse;
DROP TABLE IF EXISTS _doc_elements_stage;
DROP TABLE IF EXISTS _doc_chunks_stage;

-- 9) Vector Search endpoint + Delta Sync index (MANAGED embeddings via the custom
--    bge-m3 endpoint). `section` and (Phase 2) `corpus` are synced so the agent can
--    filter on them. Adding a synced column requires a new index: Phase 2 created
--    `doc_chunks_corpus_index` alongside the Phase-1 `doc_chunks_index` (no retrieval
--    downtime; the agent driver points at it via --vs-index-name). The job's `sync_index`
--    task (etl/sync_index.py) CREATES the index on the first run (spec as below), then
--    SYNCS it after each run, waiting for the sync and checking the indexed row count.
--    If the index is ever deleted + recreated, REDEPLOY THE AGENT afterwards: its SELECT
--    grant on the index comes from agents.deploy and doesn't carry over to a new index.
--    And never CREATE OR REPLACE doc_chunks: a new table id breaks the index's sync.
--
--    NOTE (Phase 1b): AI Search managed embeddings call the MODEL endpoint `mygpt-bge-m3`
--    directly here — NOT the `mygpt-embeddings` AI Gateway endpoint. AI Search rejects the
--    external-model (gateway) endpoint ("unsupported embedding response") in this workspace,
--    since the Custom/External Models managed-embedding path isn't available. The gateway
--    still fronts the RAG agent's embedding calls; only Vector Search stays on the model
--    endpoint. Revisit if the "AI Query for Custom Models and External Models" preview lands.
--
--    databricks vector-search-indexes create-index --profile mygpt --json '{
--      "name": "<catalog>.<schema>.doc_chunks_corpus_index",
--      "endpoint_name": "mygpt-vs",
--      "primary_key": "chunk_id",
--      "index_type": "DELTA_SYNC",
--      "delta_sync_index_spec": {
--        "source_table": "<catalog>.<schema>.doc_chunks",
--        "pipeline_type": "TRIGGERED",
--        "embedding_source_columns": [
--          {"name": "chunk_text", "embedding_model_endpoint_name": "mygpt-bge-m3"}
--        ],
--        "columns_to_sync": ["chunk_id","doc_id","source_name","corpus","section","page","elem_type","chunk_text"]
--      }
--    }'
--
-- Index-metric note: calibrate embedding_model_cutoffs (workstream C) in the SAME
-- distance space the agent uses (distance = 1 - cosine similarity returned by the index).
