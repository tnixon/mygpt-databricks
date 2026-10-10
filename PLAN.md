# MyGPT on Databricks — phased implementation roadmap

The overall guide for closing the 15 design gaps in [`README.md`](README.md#design-gaps-to-close)
step-by-step. Each phase shows distinct, building progress and gets its own research / plan /
implementation / test / validation cycle. For each phase:
**Goal · Components we build · MyGPT parts replicated · Databricks services · Dependencies · Gaps closed.**

Settled sequencing decisions: evaluation runs **measurement-first** (Phase 2); multilingual stays in
scope but **late** (Phase 6); the **current chat UI is kept as-is** through the build, with full UX
parity as the **capstone (Phase 7)**.

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

### Still open
- Redeploy the agent (v11) for the trace-experiment pin, then confirm app traces and monitor
  assessments land in `/Shared/mygpt/mygpt-rag-agent`.
- BlueSky: blocked on St. Jude's kinase PDFs and `expression_systems_list.csv`.
- llama / gptoss baselines (`mygpt_eval` matrix override via `jobs run-now`, see eval/README).

## Phase 3 — Production ingestion lane
- **Goal:** replace the hand-run SQL with a governed, distributed, resumable pipeline feeding real corpora.
- **Build:** `etl/` — Lakeflow Connect SharePoint/OneDrive → UC medallion; Spark + Jobs/Workflows (resumable, parallel) parse/chunk/embed/index; format coverage (DOCX/TXT, spreadsheets→structured tables, YouTube transcripts, Zotero); PHI scan on ingest; one-time ChromaDB → AI Search migration job.
- **Replicates:** `document_processing.py`, `video_processing.py`, `zotero_integration.py`, `dataset_management.py`, section extraction, the ingestion queue (now distributed).
- **Databricks:** Lakeflow Connect, Jobs/Workflows, Spark, `ai_parse_document`, UC (governance/PHI/lineage), Model Serving (embeddings), AI Search Delta Sync, Genie tables (for spreadsheets).
- **Depends on:** Phase 1 (embedding-model decision).
- **Closes gaps:** 4, 5, 8 (ingest), 11; plus the `etl/` medallion/PHI/governance.

## Phase 4 — Stateful backbone (Lakebase)
- **Goal:** multi-user, persistent, personalized state.
- **Build:** conversation/Q&A/sources/feedback schema; per-user upload index (Lakebase Search = vector + BM25/RRF); library self-service (create/list/delete) and public/private/group sharing; per-user RLS via UDF + session context. (History/feedback exposed in the chat UI at functional-minimum.)
- **Replicates:** the Postgres schema (Conversation/Question/Answer/Source/feedback), multi-tenant dataset ownership (`resolve_dataset_for_user`), sharing model, feedback endpoints.
- **Databricks:** Lakebase (Postgres), Lakebase Search, Unity Catalog RLS, OBO.
- **Depends on:** Phase 1 (scores to persist), Phase 3 (governed libraries to manage/share).
- **Closes gaps:** 7, 9, 14.

## Phase 5 — Multi-agent Supervisor & tools
- **Goal:** the net-new value — automatic routing, structured Q&A, domain tools.
- **Build:** `agents/` — Mosaic AI Agent Framework Supervisor routing each query to AI Search (docs) / Genie (structured) / MCPs (tools); Unity AI Gateway (FMAPI primary + external Azure AI Foundry route); Hosted MCPs porting MyGPT's domain tools (PubMed-ID, RCSB PDB) plus write-side ingest tools; Genie over the structured/spreadsheet tables.
- **Replicates:** the manual model dropdown → Supervisor; FastMCP tools (`MCP-server/mygpt_backend.py`); spreadsheet SQL querying → Genie.
- **Databricks:** Mosaic AI Agent Framework, Unity AI Gateway, Hosted MCPs, Genie, Lakebase (session/user context).
- **Depends on:** Phase 1 (retrieval as a tool), Phase 3 (structured tables), Phase 4 (user context).
- **Closes gaps:** 12; Supervisor + Genie routing (net-new).

## Phase 6 — Multilingual
- **Goal:** enable the multilingual use case (e.g. SJ Global French/Spanish).
- **Build:** per-dataset document language; query/answer translation; language-specific BM25 stemmers and rerankers; Devanagari/garbled-text parse fallback.
- **Replicates:** translation fields + `translategemma` flow, language-specific stemmers/rerankers, pypdf→PyMuPDF fallback.
- **Databricks:** FMAPI/Model Serving (translation + multilingual rerank/embeddings), AI Search, the Phase 3 ingestion lane.
- **Depends on:** Phases 1, 3.
- **Closes gaps:** 10.

## Phase 7 — Full research-assistant UX parity (capstone)
- **Goal:** bring the kept-minimal chat UI up to MyGPT's distinctive front-end.
- **Build:** `app/` client — markdown/KaTeX/PDF rendering; source PDF viewer with cited-chunk highlighting and jump-to-page/timestamp; 3D embedding visualization; pipeline flow graphs; document/section focus selector UI; full History panel (search + CSV export); feedback controls; per-answer with/without-context comparison.
- **Replicates:** `GPTHome.tsx` three-panel reader, `Plots.tsx` (3D PCA), `Flow/` graphs, `ChatHistory.tsx`, `Feedback.tsx`, highlighted-PDF workflow.
- **Databricks:** Databricks Apps (AppKit/AI DevKit); consumes the Phase 1–6 APIs.
- **Depends on:** Phases 1–6 (all data, scores, history, agents, languages to surface).
- **Closes gaps:** 6, 15; UI side of 9, 10, 13.

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
