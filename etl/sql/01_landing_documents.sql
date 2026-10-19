-- MyGPT ingest — landing layer
-- Source docs (Volume) -> FILE EXTERNAL table -> ai_parse_document downstream.
-- FILE EXTERNAL = zero-copy governed pointers to files already in the volume
-- (Beta). The SharePoint connector will later land FILE MANAGED into an
-- equivalent table; the parse/embed/index steps downstream are identical.
--
-- Prereq: FILE type Beta enabled on the workspace (Previews page). Already on in
-- the sandbox. Job parameters (databricks.yml sql_task.parameters): catalog, schema, docs_volume.

USE CATALOG IDENTIFIER(:catalog);
CREATE SCHEMA IF NOT EXISTS IDENTIFIER(:schema)
  COMMENT 'MyGPT POC — document corpus, indexes, sessions';
USE SCHEMA IDENTIFIER(:schema);

-- Raw source documents land here (seed: BioASQ biomedical + multilingual policy PDFs).
CREATE VOLUME IF NOT EXISTS IDENTIFIER(:docs_volume)
  COMMENT 'Raw source documents (FILE EXTERNAL corpus)';
-- Upload docs to /Volumes/<catalog>/<schema>/<docs_volume>/<corpus>/ (e.g. databricks fs cp -r ...).
-- Phase 2: one top-level folder = one CORPUS (biomed, health_policies_fr, health_policies_es,
-- kinase_literature). The agent filters retrieval on it via custom_inputs.corpus; see
-- `make eval-corpus` for the eval corpora layout.

-- FILE EXTERNAL table: one row per file, pointer + metadata, content read on demand.
-- Rebuilt every run (cheap: pointers only). `id` is a deterministic hash of the path so
-- doc / chunk ids are stable across runs — 02 MERGEs chunk changes keyed on them, and
-- (size_bytes, modification_time) is the file version it uses to re-parse only
-- new or changed files.
CREATE OR REPLACE TABLE documents AS
SELECT
  xxhash64(path)    AS id,
  file,                       -- FILE EXTERNAL value (governed pointer)
  path,
  size              AS size_bytes,
  modification_time,
  file.content_type AS content_type,
  -- top-level folder under the volume
  regexp_extract(path, '/Volumes/' || :catalog || '/' || :schema || '/' || :docs_volume || '/([^/]+)/', 1) AS corpus
FROM list_files('/Volumes/' || :catalog || '/' || :schema || '/' || :docs_volume || '/');  -- recursive

-- Sanity:
--   SELECT corpus, count(*) FROM <catalog>.<schema>.documents GROUP BY 1;  -- biomed 40, health_policies_fr/_es 10 each
--   SELECT ai_parse_document(file) FROM <catalog>.<schema>.documents LIMIT 1;  -- parses on demand

-- Next (02_parse_chunk_index.sql): incremental ai_parse_document -> section chunks ->
-- MERGE into <catalog>.<schema>.doc_chunks; the job's sync_index task then
-- refreshes the AI Search index over it.
