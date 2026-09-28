-- MyGPT ingest — landing layer
-- Source docs (Volume) -> FILE EXTERNAL table -> ai_parse_document downstream.
-- FILE EXTERNAL = zero-copy governed pointers to files already in the volume
-- (Beta). The SharePoint connector will later land FILE MANAGED into an
-- equivalent table; the parse/embed/index steps downstream are identical.
--
-- Prereq: FILE type Beta enabled on the workspace (Previews page). Already on in
-- the sandbox. Run against catalog `structural_biology`.

CREATE SCHEMA IF NOT EXISTS structural_biology.mygpt
  COMMENT 'MyGPT POC — document corpus, indexes, sessions';

-- Raw source documents land here (seed: BioASQ biomedical + multilingual policy PDFs).
CREATE VOLUME IF NOT EXISTS structural_biology.mygpt.docs
  COMMENT 'Raw source documents (FILE EXTERNAL corpus)';
-- Upload docs to /Volumes/structural_biology/mygpt/docs/ (e.g. databricks fs cp -r ...).

-- FILE EXTERNAL table: one row per file, pointer + metadata, content read on demand.
CREATE OR REPLACE TABLE structural_biology.mygpt.documents AS
SELECT
  monotonically_increasing_id() AS id,
  file,                       -- FILE EXTERNAL value (governed pointer)
  file.uri          AS path,
  file.size         AS size_bytes,
  file.content_type AS content_type
FROM list_files('/Volumes/structural_biology/mygpt/docs/');  -- recursive

-- Sanity:
--   SELECT count(*) FROM structural_biology.mygpt.documents;                 -- 50
--   SELECT ai_parse_document(file) FROM structural_biology.mygpt.documents LIMIT 1;  -- parses on demand

-- Next (02_parse_chunk_embed): ai_parse_document(file) -> text -> chunk ->
-- embed (databricks-gte-large-en) -> structural_biology.mygpt.doc_chunks,
-- then a Databricks AI Search index over it (see eval/ + the app retrieval wiring).
