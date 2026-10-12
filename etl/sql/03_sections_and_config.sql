-- MyGPT ingest — per-dataset scoring configuration tables.
-- Ports MyGPT's Django config (Dataset / EmbeddingModel / RerankerModel) to Delta
-- so the ResponsesAgent reads tunable coefficients, per-model distance cutoffs, and
-- reranker thresholds at query time. paper_sections is built in 02.
--
-- Seed values mirror etl/config/*.csv (the human-editable canonical source). They
-- are inlined here as VALUES so the ETL job is self-contained (no volume/CSV read
-- step). Edit both when tuning, or point a later job at the CSVs if preferred.

-- Per-dataset coefficients + retrieval settings (one row per library / corpus).
-- Phase 2 adds one row per eval corpus (dataset_name = corpus). The FR/ES policy
-- corpora run without the reranker: qnli-electra-base is English-only, matching
-- MyGPT's `global-french-bge-no-rerank` library; NLI is skipped for non-English too.
CREATE OR REPLACE TABLE structural_biology.mygpt.dataset_config AS
SELECT * FROM VALUES
  ('mygpt', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true),
  ('biomed', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true),
  ('health_policies_fr', 'bge-m3', 'structure_preserving', 1000, false, true, false,
   'cross-encoder/qnli-electra-base', 'french', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true),
  ('health_policies_es', 'bge-m3', 'structure_preserving', 1000, false, true, false,
   'cross-encoder/qnli-electra-base', 'spanish', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true),
  ('kinase_literature', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true)
AS t(dataset_name, embedding_model, chunking_method, chunksize, overlap, use_bm25, use_reranker,
      reranker, documents_language, distance_function,
      coefficient_a_Qsem, coefficient_b_Qkey, coefficient_c_Qrank,
      coefficient_x_Asem, coefficient_y_Akey, coefficient_z_Arank,
      coefficient_p_QRS, coefficient_q_ARS, HI_by_equation);

-- Per-embedding-model distance calibration (best/worst, question & answer windows).
-- bge-m3 values are BOOTSTRAP placeholders in cosine-DISTANCE space (= 1 - cosine
-- similarity). Recalibrate from reference Q/A pairs (workstream C); MyGPT's own
-- Ollama-L2 numbers do NOT transfer.
CREATE OR REPLACE TABLE structural_biology.mygpt.embedding_model_cutoffs AS
SELECT * FROM VALUES
  ('bge-m3', 0.10, 0.80, 0.10, 0.80, 'BOOTSTRAP cosine-distance placeholders; recalibrate in workstream C')
AS t(model_name, best_distance_q, worst_distance_q, best_distance_ac, worst_distance_ac, calibration_note);

-- Reranker + NLI model thresholds.
CREATE OR REPLACE TABLE structural_biology.mygpt.reranker_config AS
SELECT * FROM VALUES
  ('cross-encoder/qnli-electra-base', 0.10, 'english',
   'Query<->source relevance reranker (sigmoid). Sources below cutoff are dropped.'),
  ('cross-encoder/nli-deberta-v3-base', 0.00, 'english',
   'Answer<->source NLI entailment (contradiction/entailment/neutral); feeds the RF HI features.')
AS t(model_name, cutoff_score, language, note);

-- Sanity:
--   SELECT * FROM structural_biology.mygpt.dataset_config;
--   SELECT * FROM structural_biology.mygpt.embedding_model_cutoffs;
--   SELECT section_title, section_count FROM structural_biology.mygpt.paper_sections ORDER BY section_count DESC;
