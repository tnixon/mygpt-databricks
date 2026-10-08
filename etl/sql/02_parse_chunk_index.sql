-- MyGPT ingest — parse + section-aware chunk, then build the AI Search index.
-- Runs after 01_landing_documents.sql. Catalog: structural_biology.
--
-- Phase 1 changes vs. Phase 0:
--   * carries a running `section` per chunk (MyGPT `structure_preserving` chunking),
--     so the agent can metadata-filter retrieval to a document/section (gap 13);
--   * merges the fine-grained ai_parse_document elements into ~1000-char,
--     section-bounded chunks (MyGPT chunksize=1000, overlap=false);
--   * re-embeds via the custom `mygpt-bge-m3` Model Serving endpoint (gap 1).
-- This file is run as a task of the DABs ETL job (see databricks.yml).

-- 1) Parse each FILE and explode into document elements, tagging section headers.
--    CDF stays on so the Vector Search Delta Sync index can incrementally sync.
CREATE OR REPLACE TABLE structural_biology.mygpt.doc_elements
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
WITH parsed AS (
  SELECT id AS doc_id, path, ai_parse_document(file) AS d
  FROM structural_biology.mygpt.documents
  WHERE content_type = 'application/pdf'
  -- seed scope; drop the LIKE to include everything (e.g. multilingual policy docs)
  AND path LIKE '%/biomed/%'
),
elems AS (
  SELECT doc_id, path, idx, e
  FROM parsed
  LATERAL VIEW posexplode(from_json(to_json(d:document.elements), 'array<variant>')) t AS idx, e
),
tagged AS (
  SELECT
    doc_id,
    path,
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
    doc_id, path, idx, elem_type, page, content, header_text,
    -- running section = most recent header seen in document order
    coalesce(
      last_value(header_text) IGNORE NULLS
        OVER (PARTITION BY doc_id ORDER BY idx
              ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
      '(no section)'
    ) AS section
  FROM tagged
)
SELECT * FROM sectioned;

-- 2) Merge body elements into ~1000-char, section-bounded chunks (no overlap).
CREATE OR REPLACE TABLE structural_biology.mygpt.doc_chunks
TBLPROPERTIES (delta.enableChangeDataFeed = true) AS
WITH body AS (
  -- keep substantive body text; drop headers themselves and tiny noise
  SELECT
    doc_id, path, idx, page, section, content,
    regexp_extract(path, '([^/]+)$', 1) AS source_name
  FROM structural_biology.mygpt.doc_elements
  WHERE length(content) >= 80
    AND lower(elem_type) NOT IN ('page_header', 'page_footer', 'page_number')
),
bucketed AS (
  SELECT
    doc_id, path, source_name, idx, page, section, content,
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
  concat(doc_id, '-',
         conv(substr(md5(section), 1, 8), 16, 10), '-', bucket) AS chunk_id,
  doc_id,
  path,
  source_name,
  section,
  page,
  'section_chunk'                                                AS elem_type,
  chunk_text
FROM chunks
WHERE length(chunk_text) >= 80;

-- 3) Per-dataset section catalogue (port of MyGPT PaperSections) — powers the
--    document/section focus selector and the agent's section validation.
CREATE OR REPLACE TABLE structural_biology.mygpt.paper_sections AS
SELECT
  'mygpt'            AS dataset_name,
  section            AS section_title,
  count(*)           AS section_count
FROM structural_biology.mygpt.doc_chunks
GROUP BY section;

-- 4) Vector Search endpoint + Delta Sync index (MANAGED embeddings via the custom
--    bge-m3 endpoint). `section` is now synced so the agent can filter on it. These
--    are CLI/API, not SQL. Recreate the index (embedding model + columns changed) then sync:
--
--    NOTE (Phase 1b): AI Search managed embeddings call the MODEL endpoint `mygpt-bge-m3`
--    directly here — NOT the `mygpt-embeddings` AI Gateway endpoint. AI Search rejects the
--    external-model (gateway) endpoint ("unsupported embedding response") in this workspace,
--    since the Custom/External Models managed-embedding path isn't available. The gateway
--    still fronts the RAG agent's embedding calls; only Vector Search stays on the model
--    endpoint. Revisit if the "AI Query for Custom Models and External Models" preview lands.
--
--    databricks vector-search-indexes delete-index structural_biology.mygpt.doc_chunks_index --profile mygpt
--    databricks vector-search-indexes create-index --profile mygpt --json '{
--      "name": "structural_biology.mygpt.doc_chunks_index",
--      "endpoint_name": "mygpt-vs",
--      "primary_key": "chunk_id",
--      "index_type": "DELTA_SYNC",
--      "delta_sync_index_spec": {
--        "source_table": "structural_biology.mygpt.doc_chunks",
--        "pipeline_type": "TRIGGERED",
--        "embedding_source_columns": [
--          {"name": "chunk_text", "embedding_model_endpoint_name": "mygpt-bge-m3"}
--        ],
--        "columns_to_sync": ["chunk_id","doc_id","source_name","section","page","elem_type","chunk_text"]
--      }
--    }'
--
--    databricks vector-search-indexes sync-index structural_biology.mygpt.doc_chunks_index --profile mygpt
--
-- Index-metric note: calibrate embedding_model_cutoffs (workstream C) in the SAME
-- distance space the agent uses (distance = 1 - cosine similarity returned by the index).
