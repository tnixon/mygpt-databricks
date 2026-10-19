# MyGPT on Databricks — POC monorepo

Rebuild of St. Jude's **MyGPT** research assistant, natively on Databricks. One repo for all POC components — the app, data ingestion, the agent layer, and evaluation.

## Layout
| Folder | What | Stack |
|---|---|---|
| [`app/`](app/) | Databricks App — the MyGPT UI | Node / React / TypeScript (AppKit / AI DevKit) |
| [`etl/`](etl/) | Ingestion & medallion — Lakeflow Connect (SharePoint / OneDrive → Unity Catalog), PHI scan on ingest | Python / SQL / DABs |
| [`agents/`](agents/) | Multi-agent Supervisor + tools; FMAPI / Unity AI Gateway integration | Python |
| [`eval/`](eval/) | MLflow GenAI evaluation harness (seeded from St. Jude's MyGPT-evaluations gold data) | Python |
| [`reference/`](reference/) | Read-only **submodules** of St. Jude's source — `MyGPT` (the app) + `MyGPT-evaluations` (eval datasets) | git submodules |

## Target architecture
A single MyGPT app → a multi-agent **Supervisor** routes each query across **Databricks AI Search** (global governed docs; hybrid ANN + BM25), **Lakebase Search** (per-user uploads), and **Genie** (structured), calling models through the **Unity AI Gateway** (FMAPI primary + an external route to St. Jude's Azure AI Foundry). **Lakebase** holds session/user context and sharing. **Unity Catalog** governs data and access; observability via **MLflow + LangSmith**.

## Environment
- Sandbox: FEVM Azure workspace (CLI profile `stjude-mygpt`).
- Unity Catalog: **`structural_biology`**.
- Foundation models available in-workspace via FMAPI (Claude, Llama, Qwen, GPT-OSS, Gemma) + embeddings (GTE / BGE / Qwen3).

## Setup
Deploying to a new workspace: follow [`DEPLOYMENT.md`](DEPLOYMENT.md) (ordered steps, bundle per step, resources created).

The reference sources live in `reference/` as **git submodules**, so clone recursively:
```bash
git clone --recurse-submodules <repo-url>
# already cloned without them?  →
git submodule update --init --recursive      # or: make submodules
```
Then first-time setup (submodules + app deps):
```bash
make init
```

## Common tasks (`make`)
```bash
make dev                 # run the app locally → http://localhost:8000
make config              # install both bundles' variable-overrides.json (then edit names / ids)
make validate            # validate both bundles (platform + app)
make deploy              # deploy both bundles
make run-etl             # deploy the platform bundle, then run job mygpt_etl (any job: run-<name>)
make app-run             # deploy the app bundle and restart the app on the new code
make submodules-update   # bump reference/ submodules to latest upstream, then commit the pointers
make help                # list all targets
```
Override the workspace with `PROFILE=<name>` (default `mygpt`), e.g. `make deploy PROFILE=mygpt`.
The full deployment sequence is in [DEPLOYMENT.md](DEPLOYMENT.md#deployment-steps-at-a-glance).
Quickest path to a new workspace: edit both `variable-overrides.json` files, then
run the commands in `deploy.sh` (plain databricks CLI, one per step; set `PROFILE` at the top).

## Scope
Phase-1 POC: **grant-application use case, no PHI**, built in the sandbox. Design docs (architecture diagram, POC doc, repo insights, UX spec, design tokens) live in the FE project workspace.

## Design gaps to close
Open items from the [Repo Insights review](https://docs.google.com/document/d/1lyjx3ruaF2B5vtQOkUHOfaoKv-JLCRx_ktinAv2rXco/edit) (2026-09-18) — design points from St. Jude's current code not yet reflected in the components above. To be turned into an implementation plan.

Rows 1–8 cover the core RAG/scoring engine but are **necessary, not sufficient**: rows 9–15 come from a direct read of the `reference/MyGPT` submodule (docs + backend/MCP + frontend) and capture whole capability areas MyGPT ships today that are otherwise unaccounted for.

| # | Gap | Area | Note |
|---|---|---|---|
| 1 | Custom embeddings / rerank → Model Serving | etl / agents | St. Jude use nomic / bge-m3 / qwen3 embeddings + nomic-rerank with per-model distance cutoffs; FMAPI ships only gte/bge-large. The ingest index currently embeds with `databricks-gte-large-en` (FMAPI) — **decide** whether their exact models move to Model Serving custom endpoints behind the gateway. Diagram should show FMAPI (chat) + Model Serving (embeddings/rerank) both under the gateway. Scoring normalization also depends on per-embedding-model distance calibration (best/worst cutoffs computed from reference data) — carry this over for whichever models are used. |
| 2 | Rerank pass | app / agents | Preserve the hybrid **+ rerank** retrieval pattern; `/api/ask` does hybrid ANN+BM25 only today. |
| 3 | ARS + Hallucination Index | agents | App computes a simplified QRS (mean retrieval score). Full triad: QRS = a·semantic + b·BM25 + c·rerank; ARS = same formula on the answer; HI = 1 − (p·QRS + q·ARS)/(p+q); with per-dataset tunable weights + per-model cutoffs. MyGPT also offers an alternative HI via a trained random-forest model (22 hybrid-score features) and an NLI entailment reranker (entailment / contradiction / neutral) on the answer side. |
| 4 | ChromaDB migration | etl | One-time Job to re-embed ~60k chunks/library from St. Jude's existing ChromaDB collections into Databricks AI Search. |
| 5 | Distributed / resumable ingestion | etl | Replace the single-threaded queue with Spark + Jobs/Workflows (proof point: ~95 min to ingest a 10k-page library, not resumable today). |
| 6 | Preserve distinctive UX | app | 3D embedding viz (react-three-fiber), KaTeX / PDF / markdown rendering, flow-graph UI (@xyflow/react) — none in the current chat UI. |
| 7 | Lakebase Search + per-user RLS | app / etl | Lakebase Search = `lakebase_vector` + `lakebase_text` BM25 (RRF). True per-user isolation needs an RLS UDF + session context (UC permissions are object-level). |
| 8 | Write-side MCP / MS Graph | etl / agents | Today's MCP is read-only; the SharePoint / OneDrive ingest write tools are net-new. |
| 9 | Conversation history, Q&A persistence & feedback | app / agents | MyGPT persists conversation trees, per-answer scores, retrieved sources, and user feedback (rating + comment), with a searchable History UX and CSV export. No Q&A/feedback persistence or history surface in our plan today — map to Lakebase. |
| 10 | Multilingual retrieval & translation | etl / app | Per-dataset document language (11 langs); query + answer translation, language-specific BM25 stemmers and rerankers, and garbled/Devanagari-text fallback on parse. Only the eval data notes multilingual today. |
| 11 | Ingestion formats & connectors | etl | ETL is PDF-only today (`content_type = 'application/pdf'`). MyGPT also ingests DOCX/TXT, spreadsheets (CSV/XLSX → SQL-queryable, i.e. the Genie/structured path), YouTube transcripts (timestamped), and Zotero libraries. YouTube + Zotero are net-new connectors. |
| 12 | Domain MCP read tools | agents | Port MyGPT's FastMCP tools as Hosted MCPs: PubMed-ID extraction, RCSB PDB search / details, dataset/document listing. Gap #8 covers only the net-new write-side (SharePoint) tools. |
| 13 | Document / section focus filtering | app / etl | Query-time narrowing to specific documents or sections (e.g. Methods / Results), backed by section extraction during ingest. |
| 14 | Library self-service & sharing model | app / etl | Self-service create / list / delete of libraries and public / private / group-shared visibility with shared chat history. Gap #7 covers only per-user row enforcement, not the library CRUD + sharing UX. |
| 15 | Source PDF viewer & highlighting | app | Three-panel reader with cited chunks highlighted in the source PDF and click-to-jump to page / video timestamp (backend generates highlighted PDFs). Extends gap #6 beyond plain rendering. |
