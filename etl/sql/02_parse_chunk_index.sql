-- MyGPT ingest — parse new/changed documents into section-tagged elements (staging).
-- Runs after 01_landing_documents.sql; then etl/parse_fallback.py (Phase 6) re-extracts
-- garbled / script-less documents with PyMuPDF, and 02b_chunk_merge.sql chunks + MERGEs
-- everything into the AI Search source table. Catalog / schema: the job's `catalog` /
-- `schema` parameters.
--
-- Phase 1 changes vs. Phase 0:
--   * carries a running `section` per chunk (MyGPT `structure_preserving` chunking),
--     so the agent can metadata-filter retrieval to a document/section (gap 13);
--   * merges the fine-grained ai_parse_document elements into ~1000-char,
--     section-bounded chunks (MyGPT chunksize=1000, overlap=false);
--   * re-embeds via the custom `mygpt-bge-m3` Model Serving endpoint (gap 1).
-- Phase 2: carries `corpus` (from 01) through to the chunks + index so the agent can
--   isolate retrieval per corpus (eval corpora vs. the biomed seed).
-- Phase 2 (incremental): nothing is rebuilt from scratch.
--   * only NEW or CHANGED files (by size + modification_time, tracked in doc_parse_log)
--     go through ai_parse_document — also keeps its run-to-run variance from churning
--     chunks of unchanged documents;
--   * chunks are MERGEd into doc_chunks on a stable chunk_id (01 hashes the path), so
--     unchanged chunks emit no change-feed rows and the Delta Sync index only re-embeds
--     what changed. The job's `sync_index` task (etl/sync_index.py) then syncs the index.
--   Force a re-parse of a document by deleting its doc_parse_log row.
-- Phase 6: 02 was split at the element-staging boundary so the PyMuPDF fallback can replace
--   flagged documents' elements before chunking (steps 4-9 are now 02b, unchanged).
-- This file is run as a task of the DABs ETL job (see databricks.yml). Statements run in
-- order; the `_`-prefixed tables are per-run staging, dropped at the end of 02b.

-- Job parameters (databricks.yml sql_task.parameters): catalog, schema, docs_volume.
USE CATALOG IDENTIFIER(:catalog);
USE SCHEMA IDENTIFIER(:schema);

-- 0) Persistent tables (created on first run; existing tables keep their data).
CREATE TABLE IF NOT EXISTS doc_elements (
  doc_id BIGINT, path STRING, corpus STRING, idx INT, elem_type STRING, page INT,
  content STRING, header_text STRING, section STRING
);

-- CDF on: the Vector Search Delta Sync index syncs incrementally from the change feed.
CREATE TABLE IF NOT EXISTS doc_chunks (
  chunk_id STRING, doc_id BIGINT, path STRING, source_name STRING, corpus STRING,
  section STRING, page INT, elem_type STRING, chunk_text STRING
) TBLPROPERTIES (delta.enableChangeDataFeed = true);

-- One row per parsed document version: what ai_parse_document has already processed.
CREATE TABLE IF NOT EXISTS doc_parse_log (
  doc_id BIGINT, path STRING, corpus STRING, size_bytes BIGINT,
  modification_time TIMESTAMP, parsed_at TIMESTAMP, n_elements BIGINT
);

-- Phase 6: per parsed document version, the parse-quality check and which parser's text
-- was kept (written by etl/parse_fallback.py).
CREATE TABLE IF NOT EXISTS doc_parse_quality (
  doc_id BIGINT, path STRING, corpus STRING, size_bytes BIGINT, modification_time TIMESTAMP,
  language STRING, n_chars BIGINT, reason STRING, parser STRING, checked_at TIMESTAMP
);

-- 1) Documents in scope: PDFs in the corpus allow-list (one top-level volume folder
--    each); add new corpora here. Anything outside it is removed downstream.
CREATE OR REPLACE TABLE _docs_in_scope AS
SELECT id AS doc_id, path, corpus, size_bytes, modification_time, file
FROM documents
WHERE content_type = 'application/pdf'
  AND corpus IN ('biomed', 'health_policies_fr', 'health_policies_es', 'health_policies_hi',
                 'kinase_literature');

-- 2) New or changed documents = in scope, but this exact file version was never parsed.
CREATE OR REPLACE TABLE _docs_to_parse AS
SELECT d.*
FROM _docs_in_scope d
LEFT ANTI JOIN doc_parse_log l
  ON  d.doc_id = l.doc_id
  AND d.size_bytes = l.size_bytes
  AND d.modification_time = l.modification_time;

-- 3) Parse only those, and explode into document elements tagged with their section.
CREATE OR REPLACE TABLE _doc_elements_stage AS
WITH parsed AS (
  SELECT doc_id, path, corpus, ai_parse_document(file) AS d
  FROM _docs_to_parse
),
elems AS (
  SELECT doc_id, path, corpus, idx, e
  FROM parsed
  LATERAL VIEW posexplode(from_json(to_json(d:document.elements), 'array<variant>')) t AS idx, e
),
tagged AS (
  SELECT
    doc_id,
    path,
    corpus,
    idx,
    e:type::string                        AS elem_type,
    try_cast(e:bbox[0]:page_id AS INT)    AS page,
    trim(e:content::string)               AS content,
    -- treat ONLY titles / section headers as section boundaries (NOT page_header/
    -- page_footer/page_number, which are running furniture that would pollute sections)
    CASE WHEN lower(e:type::string) IN ('title', 'section_header')
         THEN trim(e:content::string) END AS header_text
  FROM elems
),
sectioned AS (
  SELECT
    doc_id, path, corpus, idx, elem_type, page, content, header_text,
    -- running section = most recent header seen in document order
    coalesce(
      last_value(header_text) IGNORE NULLS
        OVER (PARTITION BY doc_id ORDER BY idx
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      '(no section)'
    ) AS section
  FROM tagged
)
SELECT doc_id, path, corpus, idx, elem_type, page, content, header_text, section FROM sectioned;

-- Next: etl/parse_fallback.py (job task `parse_fallback`), then 02b_chunk_merge.sql.
