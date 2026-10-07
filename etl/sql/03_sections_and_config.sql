-- MyGPT ingest — per-dataset scoring configuration tables.
-- Ports MyGPT's Django config (Dataset / EmbeddingModel / RerankerModel) to Delta
-- so the ResponsesAgent reads tunable coefficients, per-model distance cutoffs, and
-- reranker thresholds at query time. Run as a DABs ETL job task AFTER the config
-- CSVs are synced to the config volume (bundle step: `databricks sync etl/config
-- /Volumes/structural_biology/mygpt/config`). paper_sections is built in 02.

CREATE VOLUME IF NOT EXISTS structural_biology.mygpt.config
  COMMENT 'Editable per-dataset scoring config (CSV source of truth, synced from etl/config/)';

-- Per-dataset coefficients + retrieval settings (one row per library).
CREATE OR REPLACE TABLE structural_biology.mygpt.dataset_config AS
SELECT
  dataset_name::string                 AS dataset_name,
  embedding_model::string              AS embedding_model,
  chunking_method::string              AS chunking_method,
  chunksize::int                       AS chunksize,
  overlap::boolean                     AS overlap,
  use_bm25::boolean                    AS use_bm25,
  use_reranker::boolean                AS use_reranker,
  reranker::string                     AS reranker,
  documents_language::string           AS documents_language,
  distance_function::string            AS distance_function,
  coefficient_a_Qsem::double           AS coefficient_a_Qsem,
  coefficient_b_Qkey::double           AS coefficient_b_Qkey,
  coefficient_c_Qrank::double          AS coefficient_c_Qrank,
  coefficient_x_Asem::double           AS coefficient_x_Asem,
  coefficient_y_Akey::double           AS coefficient_y_Akey,
  coefficient_z_Arank::double          AS coefficient_z_Arank,
  coefficient_p_QRS::double            AS coefficient_p_QRS,
  coefficient_q_ARS::double            AS coefficient_q_ARS,
  HI_by_equation::boolean              AS HI_by_equation
FROM read_files('/Volumes/structural_biology/mygpt/config/dataset_config.csv',
                format => 'csv', header => true, mode => 'FAILFAST');

-- Per-embedding-model distance calibration (best/worst, question & answer windows).
CREATE OR REPLACE TABLE structural_biology.mygpt.embedding_model_cutoffs AS
SELECT
  model_name::string                   AS model_name,
  best_distance_q::double              AS best_distance_q,
  worst_distance_q::double             AS worst_distance_q,
  best_distance_ac::double             AS best_distance_ac,
  worst_distance_ac::double            AS worst_distance_ac,
  calibration_note::string             AS calibration_note
FROM read_files('/Volumes/structural_biology/mygpt/config/embedding_model_cutoffs.csv',
                format => 'csv', header => true, mode => 'FAILFAST');

-- Reranker + NLI model thresholds.
CREATE OR REPLACE TABLE structural_biology.mygpt.reranker_config AS
SELECT
  model_name::string                   AS model_name,
  cutoff_score::double                 AS cutoff_score,
  language::string                     AS language,
  note::string                         AS note
FROM read_files('/Volumes/structural_biology/mygpt/config/reranker_config.csv',
                format => 'csv', header => true, mode => 'FAILFAST');

-- Sanity:
--   SELECT * FROM structural_biology.mygpt.dataset_config;
--   SELECT * FROM structural_biology.mygpt.embedding_model_cutoffs;
--   SELECT section_title, section_count FROM structural_biology.mygpt.paper_sections ORDER BY section_count DESC;
