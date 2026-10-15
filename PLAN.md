# MyGPT on Databricks — phased implementation roadmap

The overall guide for closing the 15 design gaps in [`README.md`](README.md#design-gaps-to-close)
step-by-step. Each phase shows distinct, building progress and gets its own research / plan /
implementation / test / validation cycle. For each phase:
**Goal · Components we build · MyGPT parts replicated · Databricks services · Dependencies · Gaps closed.**

Settled sequencing decisions: evaluation runs **measurement-first** (Phase 2); multilingual stays in
scope but **late** (Phase 6); the **current chat UI is kept as-is** through the build, with full UX
parity as the **capstone (Phase 7)**. The production ingestion lane (**Phase 3**) is **deferred to the
end** (decided 2026-10-13): it runs after Phase 7, so the execution order is 4 → 5 → 6 → 7 → 3. Until
then the incremental `mygpt_etl` job (volume folders → `doc_chunks` → `doc_chunks_corpus_index`) is
the ingestion lane. Phase numbers are kept stable so existing references stay valid.

### Cross-cutting UX principle
Phases 1–6 surface progress through *functional-minimum* additions to the existing chat UI
(`app/client/src/App.tsx`) — e.g. extra score badges, a plain history list, feedback buttons — only as
needed to exercise/demo each backend capability. The distinctive MyGPT front-end (PDF viewer +
highlighting, 3D embedding viz, pipeline flow graphs, polish) is deliberately deferred to Phase 7.

## Phase 0 — Foundation (DONE, increments 1–2)
Grounded RAG already works: FMAPI chat, hybrid retrieval over a Databricks AI Search index, OBO
identity split, a basic chat UI with a simplified QRS, and a hand-run PDF-only ETL. Everything below builds on this.

---

## Phase 1 — Retrieval & scoring fidelity
- **Goal:** bring the RAG engine to MyGPT-grade quality and trust metrics.
- **Build:** rerank stage; full QRS / ARS / Hallucination-Index scoring (equation + random-forest variant + NLI answer entailment); document/section focus filtering at query time; resolve the embedding/rerank model choice and port per-model distance calibration.
- **Replicates:** `get_context` / `save_answer` scoring (`analytics.py`, `HI_prediction_by_random_forest.py`, `rerank_utils.py`), cutoff-distance calibration, focused_document/section retrieval.
- **Databricks:** Model Serving (custom embeddings + rerank behind the gateway), Databricks AI Search (hybrid), FMAPI chat, MLflow custom scorers.
- **Depends on:** Phase 0.
- **Closes gaps:** 1, 2, 3, 13 (retrieval side).

**Settled design (detailed plan approved 2026-10-06):**
- **Embedding model = `bge-m3`** on a custom Model Serving endpoint, used as the **managed** embedding of
  the existing Delta Sync index — bge-m3 is prefix-free (so the Phase-0 `ai_parse_document` → Delta Sync
  ingestion pattern is preserved) *and* is the multilingual model Phase 6 (SJ Global FR/ES) will need.
- **Retrieval = a custom MLflow `ResponsesAgent`** on its own Model Serving endpoint (Mosaic AI Agent
  Framework, MLflow tracing): embed → AI Search (ANN + hybrid, document/section focus filters) → cross-encoder
  rerank (`qnli-electra-base`) → QRS → FMAPI generation → NLI (`nli-deberta-v3-base`) → ARS + Hallucination
  Index; scores + sources returned via `custom_outputs`. Node `/api/ask` becomes a thin OBO-passthrough BFF.
- **Per-dataset config** (coefficients, per-model distance cutoffs, reranker thresholds, sections) ports
  MyGPT's Django tables to **Delta tables** in `structural_biology.mygpt`.
- **Deployment = Databricks Asset Bundles (DABs)** for every component (embedding endpoint, agent endpoint,
  ETL job(s), config/section tables, app) — no hand-run deploys.
- **Fidelity = Balanced**, with higher-parity levers deferred until Phase-2 measurement shows a need:
  (a) BM25 keyword term comes from AI Search **hybrid** (fused) rescored per-source rather than a separate
  full-corpus `bm25s` index + content-dedup merge; (b) random-forest HI ships as a best-effort toggle rather
  than a retrained first-class output; (c) bge-m3 managed embeddings chosen over nomic's self-managed
  `search_document:`/`search_query:` prefix flow.

## Phase 1b — Unity AI Gateway in front of the embedding model
- **Goal:** insert a Unity AI Gateway endpoint between the embedding model and its consumers, so
  the model is reached through one governed, swappable interface rather than by hard-coded endpoint
  name — the first step toward making the Unity Gateway the standard interface in front of all
  models (anticipates the Phase 5 gateway item).
- **Build:** an external-model serving endpoint `mygpt-embeddings` (provider
  `databricks-model-serving`, task `llm/v1/embeddings`) proxying the existing `mygpt-bge-m3`
  endpoint; re-point embedding consumers at it.
- **Decisions (2026-10-07):** no separate OpenAI wrapper — `mygpt-bge-m3` already emits the OpenAI
  embeddings envelope via the OpenAI-client path (`sentence_transformers` + `task="llm/v1/embeddings"`;
  note the raw `/invocations` path returns a bare vector list, which `_parse_embeddings` already
  handles), so step 1 is a contract test; gateway uses the `databricks-model-serving` provider
  (needs a secret token with CAN QUERY on `mygpt-bge-m3`, scope `mygpt` key `gw_upstream_token`).
- **Databricks:** Mosaic AI Gateway (external models), Model Serving, Databricks AI Search.
- **Depends on:** Phase 1. **Blocks nothing** — a drop-in indirection ahead of Phase 2.

### As-built (deployed + validated 2026-10-08)
- **Gateway live:** `mygpt-embeddings` external-model endpoint (READY), proxying `mygpt-bge-m3`.
  Smoke test passes: raw `/invocations` returns 2×1024-dim vectors.
- **RAG agent re-pointed:** redeployed with `MYGPT_EMBED_ENDPOINT=mygpt-embeddings` (its
  answer-scoring `_embed` calls now traverse the gateway).
- **Validated end-to-end:** "What biomarkers predict effective temozolomide dosing in glioblastoma?"
  → grounded cited answer, **QRS 54 / ARS 86 / HI 0**, correct temozolomide sources (rerank ~0.997),
  matching the Phase-1 baseline (56 / 82 / 0). Non-zero ARS confirms the agent's `_embed` runs
  through the gateway. (A terse keyword string like "temozolomide dosing biomarkers" trips the QNLI
  reranker, which expects a question — expected behavior, not a regression.)
- **Vector Search stays DIRECT on `mygpt-bge-m3`.** AI Search **rejects** the external-model
  gateway endpoint as a managed embedding model ("unsupported embedding response") in this
  workspace — the Custom/External Models managed-embedding path isn't available. Fallback applied:
  index kept on the model endpoint. Revisit if the "AI Query for Custom Models and External Models"
  preview lands.
- **ROOT-CAUSE GOTCHA (cost hours):** the agent declares its reachable resources in
  `driver.py`; the VS index embeds the QUERY at retrieval time through its managed-embedding
  model (`mygpt-bge-m3`) **using the agent's identity**. Pointing `--embed-endpoint` at the
  gateway dropped `mygpt-bge-m3` from the declared resources → retrieval `PermissionDenied` →
  caught → silent `[]` → "couldn't find anything". FIX: declare **both** `mygpt-embeddings`
  (scoring) and `mygpt-bge-m3` (`--vs-embed-endpoint`, VS query-time embedding) as resources.
  agent.py now also surfaces the caught retrieval error in `custom_outputs.retrieve_error`
  instead of returning `[]` silently.
- **DEPLOY GOTCHA:** `databricks bundle run <job>` re-logs the **workspace** copy of the code;
  always `databricks bundle deploy` after editing `agents/**` before `bundle run`, or the driver
  logs stale code (three redeploys silently re-logged unchanged code before this was caught).

### Code map (as-built)
- `agents/gateway/{driver.py,requirements.txt,smoke_test.py}` — deploy lane + contract test for `mygpt-embeddings`.
- `databricks.yml` — `mygpt_deploy_gateway` job + secret/run-command notes.
- `agents/rag_agent/driver.py` — `--embed-endpoint` → `mygpt-embeddings` + new `--vs-embed-endpoint`
  (`mygpt-bge-m3`); BOTH declared as agent resources. `agent.py` surfaces `retrieve_error` on empty retrieval.
- `agents/embedding/smoke_test.py` — raw-vector-list + best-effort OpenAI-client contract checks.
- `etl/sql/02_parse_chunk_index.sql` — index stays on `mygpt-bge-m3`, with a note on the AI Search limitation.
- **App:** no change — it calls the `mygpt-rag-agent` endpoint, not embeddings.

### Deploy order (as-run)
`bundle deploy --var warehouse_id=<id>` → put secret `mygpt/gw_upstream_token` →
`run mygpt_deploy_gateway` → `python agents/gateway/smoke_test.py` → `run mygpt_deploy_agent` →
re-run the Phase-1 validation query. (The attempted VS re-point to the gateway is NOT part of the
final flow — it fails AI Search validation; keep the index on `mygpt-bge-m3`.)

### Watch-items
- Vector-Search-over-external-model may need the "AI Query for Custom Models and External Models"
  preview; if blocked, keep VS pointed at `mygpt-bge-m3` directly and front only the agent.
- Extra hop at query time (gateway → bge-m3); `mygpt-bge-m3` stays always-on, gateway is routing-only.
- `databricks-model-serving` token rotation is a standing operational item.

## Phase 2 — Evaluation harness (measurement-first)
- **Goal:** make quality measurable before we scale — gate every later phase on it.
- **Build:** `eval/` — MLflow 3 `genai.evaluate()` harness seeded from MyGPT-evaluations gold data; map QRS→RetrievalGroundedness, ARS→Correctness, HI→Guidelines + custom scorers (kinase extraction accuracy, structured-output validity); register datasets in UC.
- **Replicates:** the MyGPT-evaluations RAG eval harness (kinase_literature→BlueSky, health_policies→SJ Global, PubMedQA/BioASQ).
- **Databricks:** MLflow 3 GenAI eval, Unity Catalog (eval datasets), the Phase 1 scorers.
- **Depends on:** Phase 1 (scores to measure) + a seed corpus indexed (Phase 0 ETL).
- **Closes gaps:** the `eval/` component.

**Settled design (detailed plan approved 2026-10-08):**
- **Scope = the two St. Jude use cases only:** SJ Global (`health_policies`, FR + ES) and BlueSky
  (`kinase_literature`). Public benchmarks (BioASQ / PubMedQA / Open-RAG-Bench / QRS-ARS-cutoff) are deferred.
- **Guidelines-only grading** (no gold answers yet): built-in judges + code scorers. Expectations
  tables leave room for SME-labelled `expected_facts` later.
- **Corpus isolation = a `corpus` column** (the top-level docs-volume folder) on chunks + the index;
  the agent filters on `custom_inputs.corpus` (absent = all corpora, so the app is unchanged).
- **One shared, DAB-managed experiment** `/Shared/mygpt/mygpt-rag-agent` for LoggedModels, eval runs
  (linked by `model_id`), and production traces/monitoring.
- **Kinase PDFs + expression-system vocabulary come from St. Jude.**

### As-built (deployed + baselined 2026-10-09)
- **Corpus:** `docs/{biomed,health_policies_fr,health_policies_es}/` → `doc_chunks` with `corpus`
  (biomed 661 / FR 1,790 / ES 2,567 chunks; all 20 policy PDFs parse). New index
  **`doc_chunks_corpus_index`** (5,018 rows, syncs `corpus`), created *alongside* the Phase-1
  `doc_chunks_index` instead of replacing it, so there was no retrieval downtime. The old index
  and the legacy `docs/policy/` folder are now unused and can be deleted.
- **Agent v10** (LoggedModel `m-da490313…`) live at 100%. Regression query unchanged:
  QRS 54 / ARS 87 / HI 0, vs the Phase-1b 54 / 86 / 0.
- **UC eval datasets:** `eval_health_policies_{fr,es}` (210 each), `eval_kinase_literature` (282).
- **Baseline** (Sonnet, agent v10, 210 rows each; eval runs `950baa3c…` FR / `20abddbd…` ES in
  `/Shared/mygpt/mygpt-rag-agent`):

  | metric | FR | ES |
  |---|---|---|
  | retrieval_ok / focus_adherence | 1.000 / 1.000 | 1.000 / 1.000 |
  | yes_no_format | 0.986 | 0.990 |
  | retrieval_groundedness | 0.638 | 0.633 |
  | relevance_to_query | 0.990 | 1.000 |
  | expectations_guidelines | 0.757 | 0.757 |
  | mygpt_grounding (Guidelines) | 0.800 | 0.705 |
  | safety | 1.000 | 1.000 |
  | QRS / ARS / HI_eq (means) | 65.9 / 73.6 / 5.9 | 65.9 / 74.4 / 6.4 |
  | HI_eq-vs-groundedness AUROC | 0.654 | 0.605 |

  Gates in `eval/gates.yaml` are set just below these numbers. `retrieval_relevance` (~0.15) is
  informational: "does the policy mention X?" questions make chunks legitimately irrelevant
  whenever X is absent. A groundedness of ~0.63 and an AUROC of ~0.6 are the first real
  Phase-2 findings: about a third of the policy answers include unsupported detail, and MyGPT's
  HI predicts that only weakly. These are good targets for later phases (and for HI-RF retraining).
- **Production monitoring:** `retrieval_ok`, `mygpt_trust` (20%), and Safety / RelevanceToQuery /
  RetrievalGroundedness (10%) are registered on the shared experiment, sampling app traffic only
  (`mygpt.corpus = all`).
- **App** redeployed (`/api/sections` filters `dataset_name = 'mygpt'`).

### Bugs found by the eval (fixed)
- **Agent:** Sonnet 5 can return `message.content` as reasoning + text blocks. `_generate`
  passed the list through, so answer scoring sent dicts to bge-m3 (HTTP 400). This affected
  2/5 smoke rows and would have hit the app too. Fixed: keep only text blocks.
- **Trace experiment:** `agents.deploy` keeps the experiment already attached to an existing
  endpoint (the Phase-1 `/Users/<me>/mygpt_phase1`), so production traces missed the shared
  experiment and its monitors. Fixed: the driver pins `MLFLOW_EXPERIMENT_ID`. **Takes effect on
  the next `mygpt_deploy_agent`.**
- **run_eval:**
  - The endpoint serves every old version at 0% traffic, so the LoggedModel is now resolved from
    the traffic routes.
  - Post-run metrics now read the run's persisted traces, not `result_df`.
  - The endpoint root span nests the request as `{"request": …}`, which had left the coding
    matrix empty.
  - Eval parallelism is matched to the CPU-Small endpoint: 4 workers, 300 s timeout, one dataset
    at a time.
- **build_datasets:** on Databricks, `create_dataset` rejects dataset-level tags, and it needs
  `databricks-agents` in the job env.

### ETL made incremental (2026-10-12)
- `documents.id = xxhash64(path)` replaces `monotonically_increasing_id()`, which wasn't
  stable across runs.
- `02` parses only new or changed files (`doc_parse_log`, keyed on size + mtime) and MERGEs
  chunks into `doc_chunks`. The new `sync_index` task syncs `doc_chunks_corpus_index` and checks
  its row count.
- The first run after this change re-keys every chunk, so the index re-embeds everything once.
- **Rollout (2026-10-12/13):**
  - The first run's sync failed with `DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE`. An
    earlier run of the *old* ETL had `CREATE OR REPLACE`d `doc_chunks` (new Delta table id)
    after the index was built.
  - The index was deleted and recreated (9,213 rows: biomed 665 / FR 1,788 / ES 2,563 /
    kinase 4,197). That dropped the agent's SELECT grant on it, and the app returned
    `PermissionDenied` until the agent was redeployed (**v11**). Both gotchas are now in
    `etl/README.md`.
  - Steady state verified on a re-run: nothing re-parsed, MERGE 0/0/0, `sync_index` passed
    (9,213 = 9,213).
- **v11 also pins `MLFLOW_EXPERIMENT_ID`** to the shared experiment, so production traces and
  monitors now land there. Side effect: `mlflow.genai.to_predict_fn` won't copy an endpoint
  trace that's already in the current experiment, and then hands evaluate() a root-only trace
  with no RETRIEVER span. `run_eval.endpoint_predict_fn` now always copies the returned trace
  instead (guarded for evaluate's no-op trace check).

### Still open
- Confirm app traces receive monitor assessments in `/Shared/mygpt/mygpt-rag-agent`.
- BlueSky: kinase PDFs indexed (91 files, all 81 eval papers); still waiting on
  `expression_systems_list.csv` for Q11 vocabulary compliance.
- llama / gptoss baselines (`mygpt_eval` matrix override via `jobs run-now`, see eval/README).

### Gold-answer evals (built 2026-10-13; baselined 2026-10-14)
These lift the "guidelines-only" and "public benchmarks deferred" limits for the data we
already have. Details are in `eval/README.md`.
- **SME gold from St. Jude's private ground truths** (`reference/MyGPT-evaluations-private-main`,
  gitignored). These go to the restricted `eval_private` UC volume via `make eval-private`.
  - **SJ Global:** 320 SME-labelled MyGPT answers (269 Correct / 51 Incorrect) give a yes/no
    gold for every base question × policy. The Q6–10 variants inherit the yes/no label only
    (`gold_scope: label_only`).
    - New scorers: `yes_no_correct` and `gold_correctness` (Correctness against the Correct
      answers).
  - **BlueSky:** kinase ground truth, mapped onto Q11 / Q16v2 / Q17.
    - New scorers: `expression_system_correct`, `figure_ref_match` and `gold_correctness`
      (assay category).
    - **Data issue:** in ids 129–262 of St. Jude's file the `question_id` / `ground_truth`
      cells are shifted one row. The gold is realigned by question text (tag `sme_realigned`)
      and a shape check is applied. A corrected file has been requested.
- **BioASQ** (public): 163 questions (76 yes/no + 87 factoid) over `biomed`, which now holds
  all 397 BioASQ PDFs (`make eval-corpus`). Library-wide retrieval, `k = 15`.
  - New scorers: `yes_no_correct`, `factoid_match` and `gold_correctness`.
- **MyGPT parity + judge agreement** (`mygpt_eval_reference`, no endpoint calls): MyGPT's own
  answers are scored with our gold scorers.
  - SME accuracy is the parity bar.
  - The kinase results (3 configs) measure how often our scorers agree with the SMEs.
- **Baseline** (agent v12, Sonnet, full datasets, 2026-10-14). "MyGPT" is MyGPT's own recorded
  answers put through the same scorers (`mygpt_eval_reference`). For SJ Global it is the SMEs'
  grading instead, since our policy gold is derived from those answers.

  | Use case | Metric | Ours | MyGPT |
  |---|---|---|---|
  | SJ Global FR / ES | `yes_no_correct` (SME-labelled base questions) | 0.806 / 0.744 | 0.841 SME accuracy |
  | | `yes_no_correct` (Q6–10 variants, inherited) | 0.660 / 0.540 | — |
  | | `gold_correctness` | 0.759 / 0.672 | — |
  | BlueSky | `expression_system_correct` (Q11) | 0.424 | 0.47–0.64 (ctx5 / strpre / ctx15) |
  | | `figure_ref_match` (Q17) | 0.985 | 0.85–0.94 |
  | | `gold_correctness` (Q16v2) | 0.603 | 0.53–0.65 |
  | | SME accuracy, all 3 questions | — | 0.73–0.81 |
  | BioASQ | `yes_no_correct` (76) | 0.908 | 0.934 (gpt-oss-20b) |
  | | `factoid_match` (87) | 0.678 | 0.609 |
  | | `gold_correctness` | 0.519 | 0.456 |

  - **Scorer agreement with the SMEs** (on MyGPT's kinase answers): `figure_ref_match` 98–100%,
    `expression_system_correct` 83–89% (realigned slice 77–86%), `gold_correctness` 75–84%.
  - **Gaps vs MyGPT:**
    - SJ Global yes/no is below the 0.841 bar, ES more than FR. The keyword-free variants drop a
      further 15–20 points, so the answer depends on the hinted wording.
    - Kinase Q11 (expression system) is the weakest kinase question, worst on the realigned rows
      (0.371 vs 0.484 on `sme`).
  - **Dashboard:** "[mygpt] Benchmarks — MyGPT vs Databricks agent" (bundle resource
    `mygpt_benchmarks`; see `eval/README.md`). It reads the MLflow system tables, so it
    updates after every eval run. Pages: scorecard, robustness & scorer trust, trend.
  - **Gates:** set just below the baseline in `eval/gates.yaml`. The lower of FR / ES is used for
    SJ Global. The inherited slice stays informational (50 rows per language). All new gates
    pass on the baseline runs. One existing gate failed: ES `retrieval_ok` = 0.995 (1 of 210
    rows). This needs triage before `gate=true`.
- **Still out of scope:**
  - PubMedQA (no PDFs).
  - QRS-ARS-cutoff (mostly PubMedQA, so cutoff recalibration is still open).
  - Open-RAG-Bench (PDFs not downloaded).
- **Rollout:**
  1. `bundle deploy` (creates the `eval_private` volume).
  2. `make eval-private` and `make eval-corpus`, then `bundle run mygpt_etl`.
  3. `bundle run mygpt_eval_build_datasets`.
  4. `bundle run mygpt_eval_reference`.
  5. `mygpt_eval` smoke (`max_rows=5`), then a full run.
  6. Set the gates (done), then re-run with `gate=true`.

## Phase 4 — Stateful backbone (Lakebase)
- **Goal:** multi-user, persistent, personalized state.
- **Build:** conversation/Q&A/sources/feedback schema; per-user upload index (Lakebase Search = vector + BM25/RRF); library self-service (create/list/delete) and public/private/group sharing; per-user RLS via UDF + session context. (History/feedback exposed in the chat UI at functional-minimum.)
- **Replicates:** the Postgres schema (Conversation/Question/Answer/Source/feedback), multi-tenant dataset ownership (`resolve_dataset_for_user`), sharing model, feedback endpoints.
- **Databricks:** Lakebase (Postgres), Lakebase Search, Unity Catalog RLS, OBO.
- **Depends on:** Phase 1 (scores to persist); the Phase-2 incremental ETL (a library = a volume-folder corpus + its `dataset_config` row). Phase 3 later adds SharePoint-sourced libraries under the same model.
- **Closes gaps:** 7, 9, 14.

**Settled design (plan approved 2026-10-13):**
- **Libraries** are one Lakebase table with two kinds:
  - **corpus**: the governed AI Search corpora, seeded as public and admin-owned. "All
    governed documents" is the default.
  - **upload**: user-created libraries, indexed in Lakebase.
- **Upload index** = pgvector (bge-m3, HNSW cosine) plus full-text (`tsvector`/`ts_rank_cd`),
  fused with RRF.
  - The **app** runs this search under RLS and hands the hits to the agent as
    `custom_inputs.sources`.
  - The agent still reranks and scores (QRS/ARS/HI) and generates the answer, so the agent
    never touches Lakebase.
- **RLS = session context.**
  - The app connects as its service principal, which owns the tables, and every table has
    `FORCE ROW LEVEL SECURITY`.
  - Each request runs in one transaction that sets `mygpt.user` / `mygpt.groups`.
  - Policies are inlined on `libraries`. Child tables inherit them through RLS-filtered
    `IN (SELECT id …)` subqueries.
  - With no context set, only public rows are visible.
- **Groups** = Databricks workspace groups, read from SCIM `/Me` with the user's token (cached
  for 5 minutes).
- **Upload ingest** = the `mygpt_upload_ingest` job, triggered by the app, with one active run
  per library.
  - It uses `ai_parse_document` and the same section-chunk SQL as `02`.
  - It embeds via the `mygpt-embeddings` gateway in driver-side batches. This replaces the
    planned `ai_query` over the gateway: whether that works isn't proven, and the agent already
    uses this REST path.
  - It commits one transaction per document, so a failed run can be re-run without redoing
    finished files.
- **Shared chat history** is opt-in per library (`share_history`). It applies only to
  non-private libraries, and other members can read the history but not write to it.
- **Feedback** is one row per (answer, user). It's also logged as a `user_rating` HUMAN
  assessment on the agent's MLflow trace, using the agent's new `custom_outputs.trace_id`.

### As-built (deployed + validated 2026-10-13)
- **Deployed:**
  - Lakebase project `mygpt`: `production` branch, `primary` endpoint, schema `mygpt` created
    by the app SP through migrations `001`–`003`.
  - `uploads` volume and `mygpt_upload_ingest` job (id 352016610803853).
  - Agent v12 at 100%.
  - The app, now bound to `app/databricks.yml` with 11 resources.
- **Validated live:**
  - Identity and SCIM groups; the 5 seeded libraries; per-library sections.
  - Regression query through the app: QRS 54 / ARS 86 / HI 0, persisted to history.
  - 👍 feedback is saved and written as a `user_rating` trace assessment.
  - Upload of 2 FR policy PDFs → the ingest job indexed them (29 + 49 chunks) → the
    upload-library Q&A is answered from only those documents.
  - The app's UC resources grant `USE CATALOG` / `USE SCHEMA` themselves, and the SP may create
    the `vector` extension. Both earlier watch-items are closed.
- **Fixed during deploy:**
  - `iam.current-user:read` is a default app scope and is rejected if listed in
    `user_api_scopes`.
  - The SDK resolves `config.host` lazily, so uploads now use `await config.getHost()`.
  - `.cache()` (PERSIST) isn't allowed on serverless, so the ingest job no longer calls it.
  - A failed run left documents stuck in `parsing`. They're now marked `failed`, and there's a
    new `POST /api/libraries/:id/retry` endpoint plus a UI button.
  - The SDK's `apiClient.request` turns a top-level JSON array into `{"0": …}`, and
    `mygpt-embeddings` returns a bare `[[…]]`. `parseEmbedding` handles both.
  - Keyword search had used AND-ed `websearch_to_tsquery`, which never matched a
    natural-language question. It now uses OR-matching plus **Okapi BM25 in SQL** (IDF over the
    library), because Lakebase has no `pg_search`: only `vector`, `lakebase_vector` and
    `pg_trgm` are available.
  - Upload libraries now take their scoring config from their language (`french` →
    `health_policies_fr`, `spanish` → `health_policies_es`, which skip the English-only
    reranker/NLI). Migration `003` backfills existing libraries.
- **Open finding (Phase 6 / eval):** "objectifs du plan … en Algérie" fails the same way on the
  upload library and on the governed FR corpus (AI Search). Both retrieve the methodology chunk
  rather than the objectives list. This is general French retrieval quality, not specific to
  uploads. Add upload libraries to the Phase-2 eval before tuning RRF / BM25 weights.
- **Deploy gotcha:** `bundle run app` deploys the last-*synced* source. Run the app bundle's
  own `bundle deploy` first (a root deploy doesn't sync `app/`).

### Code map
- **App server:**
  - `app/server/db/{migrations,migrate,context,store}.ts`: schema, RLS, seed, and RLS-scoped
    data access.
  - `identity.ts` (email + groups), `ingest.ts` (job trigger and reconcile), `volumes.ts`
    (uploads volume).
  - `retrieval/{rrf,lakebaseSearch}.ts`, `routes/{libraries,ask,conversations}.ts`.
- **Tests:**
  - `server/db/rls.test.ts`: 14 RLS cases on in-process Postgres (PGlite + pgvector), run as
    a NOSUPERUSER owner like the SP.
  - `retrieval/lakebaseSearch.test.ts`: runs the hybrid SQL.
  - `validation.test.ts`, `rrf.test.ts`.
- **App client:** `app/client/src/{api.ts,MessageBubble.tsx,LibrariesPage.tsx,HistoryPage.tsx}`.
  `App.tsx` gets a library selector, "New conversation", 👍/👎 feedback, and `/libraries` +
  `/history` routes.
- **Agent:** `agents/rag_agent/agent.py` adds `sources_from_inputs` (provided sources run
  through the same RETRIEVER span) and `custom_outputs.trace_id`. Tests are in
  `test_provided_sources.py`, which is part of `make eval-test`.
- **Ingest job:** `etl/upload_ingest.py`.
- **Bundles:**
  - Root `databricks.yml` adds `postgres_projects.mygpt`, `volumes.uploads` and
    `jobs.mygpt_upload_ingest`.
  - `app/databricks.yml` adds the postgres / embeddings / volume / table / job / warehouse /
    experiment resources (`iam.current-user:read` for SCIM `/Me` is a default app scope; it can't be listed explicitly).
- **Stateless fallback:** without `LAKEBASE_ENDPOINT` (local dev) the app serves only the
  default governed library: no history, feedback or uploads.

### Deploy order
1. `databricks bundle deploy -p mygpt --var warehouse_id=<id>` at the repo root. This creates
   the Lakebase project, the uploads volume and the ingest job, and syncs the agent code.
2. `databricks bundle run mygpt_deploy_agent -p mygpt --var warehouse_id=<id>`. This deploys
   the agent version that accepts `custom_inputs.sources` and returns `trace_id`.
3. In `app/`, once only: `databricks bundle deployment bind app st-jude-mygpt -p mygpt`.
4. In `app/`: `databricks bundle deploy -p mygpt --var warehouse_id=<id> --var upload_job_id=<id>
   --var experiment_id=2750091371639406`, then `databricks bundle run app -p mygpt`. The job id
   comes from `databricks bundle summary -p mygpt` at the repo root.
5. On first start the app SP creates the `mygpt` schema (migrations `001`/`002`); check the app
   logs for `[lakebase] schema mygpt ready`.

### Watch-items
- Whether the ingest job's run-as identity (the deployer, i.e. the Lakebase project owner) can
  write the SP-owned tables. It should, through `databricks_superuser`; otherwise add grants.
- Whether the app SP may `CREATE EXTENSION vector` (migration `001`). If startup can't reach
  Lakebase or a migration fails, the app logs `[lakebase] UNAVAILABLE` and keeps serving in
  stateless mode.
- `upload_ingest.py` reads every file in the library's folder with `read_files` and then joins
  down to the pending documents. That's fine for small libraries; add a path filter if they
  grow large.
- pgvector HNSW filters per library *after* the ANN scan. For very large shared indexes,
  consider `hnsw.iterative_scan` or per-library partial indexes.

## Phase 5 — Multi-agent Supervisor & tools
- **Goal:** the net-new value — automatic routing, structured Q&A, domain tools.
- **Build:** `agents/` — Mosaic AI Agent Framework Supervisor routing each query to AI Search (docs) / Genie (structured) / MCPs (tools); Unity AI Gateway (FMAPI primary + external Azure AI Foundry route); Hosted MCPs porting MyGPT's domain tools (PubMed-ID, RCSB PDB) plus write-side ingest tools; Genie over the structured/spreadsheet tables.
- **Replicates:** the manual model dropdown → Supervisor; FastMCP tools (`MCP-server/mygpt_backend.py`); spreadsheet SQL querying → Genie.
- **Databricks:** Mosaic AI Agent Framework, Unity AI Gateway, Hosted MCPs, Genie, Lakebase (session/user context).
- **Depends on:** Phase 1 (retrieval as a tool), Phase 4 (user context); a seed structured table for Genie, landed by a minimal loader in Phase 5. Spreadsheet→table ingestion at scale arrives with Phase 3.
- **Closes gaps:** 12; Supervisor + Genie routing (net-new).

## Phase 6 — Multilingual
- **Goal:** enable the multilingual use case (e.g. SJ Global French/Spanish).
- **Build:** per-dataset document language; query/answer translation; language-specific BM25 stemmers and rerankers; Devanagari/garbled-text parse fallback.
- **Replicates:** translation fields + `translategemma` flow, language-specific stemmers/rerankers, pypdf→PyMuPDF fallback.
- **Databricks:** FMAPI/Model Serving (translation + multilingual rerank/embeddings), AI Search, the `mygpt_etl` ingestion lane.
- **Depends on:** Phases 1, 2. The Devanagari/garbled-text parse fallback goes into `etl/sql/02_parse_chunk_index.sql`.
- **Closes gaps:** 10.

## Phase 7 — Full research-assistant UX parity (capstone)
- **Goal:** bring the kept-minimal chat UI up to MyGPT's distinctive front-end.
- **Build:** `app/` client — markdown/KaTeX/PDF rendering; source PDF viewer with cited-chunk highlighting and jump-to-page/timestamp; 3D embedding visualization; pipeline flow graphs; document/section focus selector UI; full History panel (search + CSV export); feedback controls; per-answer with/without-context comparison.
- **Replicates:** `GPTHome.tsx` three-panel reader, `Plots.tsx` (3D PCA), `Flow/` graphs, `ChatHistory.tsx`, `Feedback.tsx`, highlighted-PDF workflow.
- **Databricks:** Databricks Apps (AppKit/AI DevKit); consumes the Phase 1–6 APIs.
- **Depends on:** Phases 1, 2, 4–6 (all data, scores, history, agents, languages to surface). Jump-to-timestamp for video sources lands once Phase 3 ingests YouTube transcripts.
- **Closes gaps:** 6, 15; UI side of 9, 10, 13.

## Phase 3 — Production ingestion lane (DEFERRED — runs last)
Deferred on 2026-10-13; it now runs after Phase 7. It takes over what the interim `mygpt_etl` job
stands in for in Phases 4–6: real SharePoint sources, structured spreadsheet tables, and the
language-aware parse path.
- **Goal:** replace the hand-run SQL with a governed, distributed, resumable pipeline feeding real corpora.
- **Build:** `etl/` — Lakeflow Connect SharePoint/OneDrive → UC medallion; Spark + Jobs/Workflows (resumable, parallel) parse/chunk/embed/index; format coverage (DOCX/TXT, spreadsheets→structured tables, YouTube transcripts, Zotero); PHI scan on ingest; one-time ChromaDB → AI Search migration job.
- **Replicates:** `document_processing.py`, `video_processing.py`, `zotero_integration.py`, `dataset_management.py`, section extraction, the ingestion queue (now distributed).
- **Databricks:** Lakeflow Connect, Jobs/Workflows, Spark, `ai_parse_document`, UC (governance/PHI/lineage), Model Serving (embeddings), AI Search Delta Sync, Genie tables (for spreadsheets).
- **Depends on:** Phase 1 (embedding-model decision).
- **Closes gaps:** 4, 5, 8 (ingest), 11; plus the `etl/` medallion/PHI/governance.

---

## Phase 1 — implementation notes (as-built, 2026-10-07)
Phase 1 is **built, deployed, and validated end-to-end** in the FEVM sandbox (CLI profile `mygpt`,
catalog `structural_biology.mygpt`). Everything is deployed via DABs (platform bundle at repo-root
`databricks.yml`) + the app bundle (`app/databricks.yml`). Gaps 1, 2, 3, and 13 (retrieval side) are closed.

### What is live
| Component | Resource |
|---|---|
| Custom embeddings | `mygpt-bge-m3` Model Serving endpoint (BAAI/bge-m3, GPU small, **always-on**) |
| Vector index | `structural_biology.mygpt.doc_chunks_index` — DELTA_SYNC, bge-m3 **managed** embeddings, 662 section-aware chunks, `section` synced |
| Retrieval+scoring agent | `mygpt-rag-agent` Model Serving endpoint (MLflow `ResponsesAgent`, scale-to-zero) |
| Config/section tables | `dataset_config`, `embedding_model_cutoffs`, `reranker_config`, `paper_sections` |
| App | `st-jude-mygpt` (RUNNING) — thin OBO BFF → agent; ARS/HI badges + section focus |

### Code map
- `agents/rag_agent/agent.py` — **self-contained** `ResponsesAgent` (scoring + config + RF inlined; see below). Pipeline: retrieve (AI Search hybrid REST) → rerank (`qnli-electra-base`) → QRS → generate (FMAPI chat REST) → NLI (`nli-deberta-v3-base`) → ARS + HI. Scores/sources in `custom_outputs`.
- `agents/rag_agent/scoring.py` (+ `test_scoring.py`, 10 passing golden tests) — the verified reference the agent's inlined math mirrors.
- `agents/rag_agent/{config.py,hi_random_forest.py,requirements.txt,driver.py}`, `agents/rag_agent/artifacts/random_forest_hybrid_model_base.pkl`.
- `agents/embedding/{driver.py,requirements.txt,smoke_test.py}`.
- `etl/sql/02_parse_chunk_index.sql` (section-aware ~1000-char chunking, bge-m3 index), `etl/sql/03_sections_and_config.sql` (inline config seed), `etl/config/*.csv` (canonical editable copy).
- `app/server/server.ts` (`/api/ask` → agent, `/api/sections`), `app/client/src/App.tsx` (QRS/ARS/HI badges, per-source rerank/NLI, section selector), `app/app.yaml`, `app/databricks.yml`.

### Deviations discovered during deployment (and why)
- **Agent is one self-contained file.** MLflow validates a code-based model by importing `agent.py` **in isolation** at log time, so sibling-module imports (`scoring`/`config`/`hi_random_forest`) and `code_paths` don't resolve. Fix: inline them into `agent.py`; keep `scoring.py` + tests as the reference.
- **Agent talks to VS / FMAPI over REST** (`WorkspaceClient().api_client.do(...)`), not the `VectorSearchClient` or `get_open_ai_client()` SDK helpers. The injected serving credentials work reliably over REST; the SDK clients hit credential-discovery / `databricks-sdk[openai]` gaps inside Model Serving. Embeddings use the same REST invocations path.
- **MLflow runs a full validation `predict` at log time.** The agent-deploy job env therefore needs the serving libs (`torch`, `sentence-transformers`, `scikit-learn`) and the live endpoints must be reachable; `_retrieve` also degrades to `[]` on error so the validation predict never hard-fails.
- **bge-m3 must be always-on** (`scale_to_zero_enabled=false`): Vector Search embeds the **query** through it at query time, which errors if the endpoint is cold.
- **Config seed is inlined** in `03_sections_and_config.sql` (serverless `spark_python_task` has no `__file__`, and the CSV-via-volume sync was brittle); `etl/config/*.csv` remains the editable source of truth. Driver scripts take `--base-dir ${workspace.file_path}` for the same `__file__` reason.
- **App deploy** uses `databricks apps deploy <name> --source-code-path <workspace path>` (the app already existed from Phase 0 under a teammate's bundle, so `bundle deploy` 409s on create). The Apps platform builds the Node app on deploy (no `dist/` upload needed).

### Validated
Live query ("temozolomide dosing biomarkers") returned a grounded, cited answer with **QRS 56 / ARS 82 / HI 0**, 4 sources carrying vector + rerank scores, section labels, and NLI entailment tags. Section-focus filtering confirmed (exact-match `filters_json` on `section`). Scoring math covered by `test_scoring.py` (10/10).

### Known limitations / follow-ups (not blocking)
- **HI-RF returns `None`** — the pickled random forest won't load under the serving image's scikit-learn (version drift). Equation-HI is primary and works; retraining the RF on Databricks is the documented higher-parity step.
- **Distance cutoffs are bootstrap placeholders** (`embedding_model_cutoffs`, cosine-distance space), so `vector_score` can overshoot 1.0. Recalibrate from reference Q/A pairs (workstream C) — ties into Phase 2.
- **Section-focus UI selector** needs `MYGPT_WAREHOUSE_ID` set in `app.yaml` (powers `/api/sections`); the focus mechanism works regardless when a section is passed.
- **OBO-through-agent** not yet exercised (Phase-1 corpus is non-PHI); the app forwards `x-forwarded-access-token`, to validate against RLS in Phase 4.
- **Balanced-fidelity levers** still open per the Phase-1 "Settled design" note: full-corpus bm25s + content-dedup merge, first-class RF-HI, nomic self-managed embeddings.
