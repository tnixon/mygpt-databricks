# agents — MyGPT multi-agent layer

The **Supervisor** agent (Mosaic AI Agent Framework) that routes each query to the right specialist — **Databricks AI Search** (governed docs), **Genie** (structured), **Hosted MCPs** (tools) — and calls models through the **Unity AI Gateway** (FMAPI primary + external Azure AI Foundry route). Also carries the QRS / ARS / Hallucination-style scoring the app surfaces. AI side. Python.

| Folder | Endpoint / object | What | Deployed by |
|---|---|---|---|
| `embedding/` | `mygpt-bge-m3` | bge-m3 embeddings (GPU, always-on) | `mygpt_deploy_embedding` |
| `gateway/` | `mygpt-embeddings`, `mygpt-chat` | Unity AI Gateway endpoints (`--task embeddings\|chat`); chat adds usage tracking, an inference table, a per-user rate limit and the optional Azure AI Foundry fallback | `mygpt_deploy_gateway`, `mygpt_deploy_chat_gateway` |
| `gateway/` (`--task translate`) | `mygpt-translate` | Phase 6: query/answer translation for non-English corpora (→ `databricks-gemma-3-12b`) | `mygpt_deploy_translate_gateway` |
| `rag_agent/` | `mygpt-rag-agent` | (translate) → retrieve → rerank (qnli EN / mmarco multilingual) → QRS → generate → NLI (EN) → ARS (stemmed answer BM25) / HI | `mygpt_deploy_agent` |
| `tools/` | `structural_biology.mygpt_tools.*` | Domain tools as UC functions, served by managed MCP: PubMed (ids, summaries), RCSB PDB (by gene, full text, details), governed corpora | `mygpt_deploy_tools` |
| `genie/` | Genie space "MyGPT structured data" | Kinome, kinase papers, document catalogue | `mygpt_deploy_genie` |
| `supervisor/` | `mygpt-supervisor` | The app's backend: a tool-calling router over all of the above. It picks which of the user's selected libraries to search (one RAG call each) and merges them; it never changes libraries | `mygpt_deploy_supervisor` |

Tracing: MLflow (the shared experiment) is the system of record for eval, monitoring and app
feedback. When the `mygpt/langsmith_api_key` secret exists, the drivers also turn on a LangSmith
mirror of the same spans (`LANGSMITH_*` endpoint env vars; see DEPLOYMENT.md step 4). The
Supervisor passes its retrieve run's headers to the RAG agent (`custom_inputs.langsmith_parent`),
so each question is one LangSmith tree. Both agents return `custom_outputs.langsmith_run_id`.
For local runs, put the key in `.env` (see `.env.example`).

Tests: `make eval-test` (no workspace). Live checks: `agents/gateway/smoke_test.py [--task chat]`,
`agents/supervisor/smoke_test.py`. Design and deploy order: `PLAN.md` § Phase 5.
