-- MyGPT ingest — parse + chunk, then build the AI Search (Vector Search) index.
-- Runs after 01_landing_documents.sql. Catalog: structural_biology.

-- 1) Parse each FILE with ai_parse_document (returns VARIANT) and explode its
--    document elements into ~paragraph-level chunks, keeping citation metadata.
--    CDF is enabled so the Vector Search Delta Sync index can incrementally sync.
CREATE OR REPLACE TABLE structural_biology.mygpt.doc_chunks
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
)
SELECT
  concat(doc_id, '-', idx)                 AS chunk_id,
  doc_id,
  path,
  regexp_extract(path, '([^/]+)$', 1)      AS source_name,
  try_cast(e:bbox[0]:page_id AS INT)       AS page,
  e:type::string                           AS elem_type,
  e:content::string                        AS chunk_text
FROM elems
WHERE length(e:content::string) >= 80;     -- drop tiny headers/noise

-- 2) Vector Search endpoint + Delta Sync index (managed embeddings).
--    These are CLI/API, not SQL:
--
--    databricks vector-search-endpoints create-endpoint mygpt-vs STANDARD --profile <P>
--
--    databricks vector-search-indexes create-index --profile <P> --json '{
--      "name": "structural_biology.mygpt.doc_chunks_index",
--      "endpoint_name": "mygpt-vs",
--      "primary_key": "chunk_id",
--      "index_type": "DELTA_SYNC",
--      "delta_sync_index_spec": {
--        "source_table": "structural_biology.mygpt.doc_chunks",
--        "pipeline_type": "TRIGGERED",
--        "embedding_source_columns": [
--          {"name": "chunk_text", "embedding_model_endpoint_name": "databricks-gte-large-en"}
--        ],
--        "columns_to_sync": ["chunk_id","doc_id","source_name","page","elem_type","chunk_text"]
--      }
--    }'
--
--    databricks vector-search-indexes sync-index structural_biology.mygpt.doc_chunks_index --profile <P>
--
-- 3) Query (hybrid ANN + BM25) — use the REST API (the Go CLI mis-parses the 200 response):
--    POST /api/2.0/vector-search/indexes/structural_biology.mygpt.doc_chunks_index/query
--    { "columns":["source_name","page","chunk_text"], "query_text":"...", "num_results":5, "query_type":"HYBRID" }
--
-- The app's /api/ask route calls this endpoint, builds a grounded prompt, and
-- generates the answer via the serving plugin (FMAPI).
