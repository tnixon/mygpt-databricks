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
make validate            # validate the app before deploy
make deploy              # deploy to the workspace
make submodules-update   # bump reference/ submodules to latest upstream, then commit the pointers
make help                # list all targets
```
Override the workspace with `PROFILE=<name>` (default `stjude-mygpt`), e.g. `make deploy PROFILE=stjude-mygpt`.

## Scope
Phase-1 POC: **grant-application use case, no PHI**, built in the sandbox. Design docs (architecture diagram, POC doc, repo insights, UX spec, design tokens) live in the FE project workspace.
