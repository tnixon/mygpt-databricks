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

## Phase 2 — Evaluation harness (measurement-first)
- **Goal:** make quality measurable before we scale — gate every later phase on it.
- **Build:** `eval/` — MLflow 3 `genai.evaluate()` harness seeded from MyGPT-evaluations gold data; map QRS→RetrievalGroundedness, ARS→Correctness, HI→Guidelines + custom scorers (kinase extraction accuracy, structured-output validity); register datasets in UC.
- **Replicates:** the MyGPT-evaluations RAG eval harness (kinase_literature→BlueSky, health_policies→SJ Global, PubMedQA/BioASQ).
- **Databricks:** MLflow 3 GenAI eval, Unity Catalog (eval datasets), the Phase 1 scorers.
- **Depends on:** Phase 1 (scores to measure) + a seed corpus indexed (Phase 0 ETL).
- **Closes gaps:** the `eval/` component.

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
