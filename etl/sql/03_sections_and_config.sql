-- MyGPT ingest — per-dataset scoring configuration tables.
-- Ports MyGPT's Django config (Dataset / EmbeddingModel / RerankerModel) to Delta
-- so the ResponsesAgent reads tunable coefficients, per-model distance cutoffs, and
-- reranker thresholds at query time. paper_sections is built in 02.
--
-- Seed values mirror etl/config/*.csv (the human-editable canonical source). They
-- are inlined here as VALUES so the ETL job is self-contained (no volume/CSV read
-- step). Edit both when tuning, or point a later job at the CSVs if preferred.

-- Per-dataset coefficients + retrieval settings (one row per library / corpus).
-- Phase 2 adds one row per eval corpus (dataset_name = corpus). Phase 6 (multilingual):
--   * non-English corpora rerank with MyGPT's multilingual default,
--     mmarco-mMiniLMv2-L12-H384-v1 (qnli-electra-base is English-only); NLI stays English-only;
--   * translate_query: the agent translates the question into documents_language for
--     retrieval/rerank, and the answer likewise for ARS scoring (MyGPT's translategemma flow);
--   * answer_bm25: ARS keyword term = stemmed BM25 of the answer over the retrieved contexts;
--   * default_<language> rows are the configs for upload libraries in that language
--     (app/server/db/store.ts configDatasetFor). 'unknown' reranks multilingually, no translation.
CREATE OR REPLACE TABLE structural_biology.mygpt.dataset_config AS
SELECT * FROM VALUES
  ('mygpt', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, false, true),
  ('biomed', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, false, true),
  ('health_policies_fr', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'french', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('health_policies_es', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'spanish', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('health_policies_hi', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'hindi', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('kinase_literature', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/qnli-electra-base', 'english', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, false, true),
  ('default_portuguese', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'portuguese', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_german', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'german', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_italian', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'italian', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_dutch', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'dutch', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_arabic', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'arabic', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_hindi', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'hindi', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_hungarian', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'hungarian', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, true, true),
  ('default_unknown', 'bge-m3', 'structure_preserving', 1000, false, true, true,
   'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 'unknown', 'cosine',
   4.0, -4.0, -1.0, 5.0, -2.0, 0.0, 1.0, 2.0, true, false, true)
AS t(dataset_name, embedding_model, chunking_method, chunksize, overlap, use_bm25, use_reranker,
      reranker, documents_language, distance_function,
      coefficient_a_Qsem, coefficient_b_Qkey, coefficient_c_Qrank,
      coefficient_x_Asem, coefficient_y_Akey, coefficient_z_Arank,
      coefficient_p_QRS, coefficient_q_ARS, HI_by_equation, translate_query, answer_bm25);

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
  ('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', 0.05, 'multilingual',
   'Query<->source relevance reranker for non-English documents (sigmoid); MyGPT non-English default.'),
  ('cross-encoder/nli-deberta-v3-base', 0.00, 'english',
   'Answer<->source NLI entailment (contradiction/entailment/neutral); feeds the RF HI features.')
AS t(model_name, cutoff_score, language, note);

-- Sanity:
--   SELECT * FROM structural_biology.mygpt.dataset_config;
--   SELECT * FROM structural_biology.mygpt.embedding_model_cutoffs;
--   SELECT section_title, section_count FROM structural_biology.mygpt.paper_sections ORDER BY section_count DESC;
