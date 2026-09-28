# MyGPT on Databricks — POC monorepo

Rebuild of St. Jude's **MyGPT** research assistant, natively on Databricks. One repo for all POC components — the app, data ingestion, the agent layer, and evaluation.

## Layout
| Folder | What | Stack |
|---|---|---|
| [`app/`](app/) | Databricks App — the MyGPT UI | Node / React / TypeScript (AppKit / AI DevKit) |
| [`etl/`](etl/) | Ingestion & medallion — Lakeflow Connect (SharePoint / OneDrive → Unity Catalog), PHI scan on ingest | Python / SQL / DABs |
| [`agents/`](agents/) | Multi-agent Supervisor + tools; FMAPI / Unity AI Gateway integration | Python |
| [`eval/`](eval/) | MLflow GenAI evaluation harness (seeded from St. Jude's MyGPT-evaluations gold data) | Python |

## Target architecture
A single MyGPT app → a multi-agent **Supervisor** routes each query across **Databricks AI Search** (global governed docs; hybrid ANN + BM25), **Lakebase Search** (per-user uploads), and **Genie** (structured), calling models through the **Unity AI Gateway** (FMAPI primary + an external route to St. Jude's Azure AI Foundry). **Lakebase** holds session/user context and sharing. **Unity Catalog** governs data and access; observability via **MLflow + LangSmith**.

## Environment
- Sandbox: FEVM Azure workspace (CLI profile `stjude-mygpt`).
- Unity Catalog: **`structural_biology`**.
- Foundation models available in-workspace via FMAPI (Claude, Llama, Qwen, GPT-OSS, Gemma) + embeddings (GTE / BGE / Qwen3).

## Getting started (app)
```bash
cd app
npm run dev                                   # local dev → http://localhost:8000
databricks apps validate --profile <profile>  # validate before deploy
databricks apps deploy  --profile <profile>   # deploy to the workspace
```

## Scope
Phase-1 POC: **grant-application use case, no PHI**, built in the sandbox. Design docs (architecture diagram, POC doc, repo insights, UX spec, design tokens) live in the FE project workspace.
