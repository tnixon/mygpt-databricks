# MyGPT — Databricks resource inventory

How these resources connect: see the diagrams in **[ARCHITECTURE.md](ARCHITECTURE.md)**, which use the same names.

Every Databricks resource that MyGPT deploys or uses. Names come from the bundle variables. The
values shown are the current deployment's, taken from `.databricks/bundle/default/variable-overrides.json`
and `app/.databricks/bundle/default/variable-overrides.json`, which `make config` installs from the
`variable-overrides.example.json` files. Where a variable isn't overridden, its default in
`databricks.yml` applies. IDs come from the bundles' deployment state (`databricks bundle summary -p mygpt`).

- **Workspace:** `https://adb-7405616422000820.0.azuredatabricks.net` (CLI profile `mygpt`)
- **Platform bundle:** `st-jude-mygpt-platform` (`databricks.yml`): data, agents, evals
- **App bundle:** `st-jude-mygpt` (`app/databricks.yml`): the chat app

"Owner" says how each resource comes to exist:
- **bundle:** declared in a bundle; `bundle deploy` creates it.
- **job:** a bundle job creates it when it runs.
- **app:** the app creates it at runtime.
- **pre-existing:** created outside the bundles (by an admin, `deploy.sh`, or Databricks itself).

---

## 1. Compute and infrastructure

| Name / ID | Variable | Type | Owner | Role |
|---|---|---|---|---|
| `79388385e899d08b` | `warehouse_id` | SQL warehouse (serverless) | pre-existing | Runs the ETL and tools SQL tasks. Backs the dashboard, the Genie space and the Supervisor's Genie calls. The app uses it for the section catalogue, and the RAG agent can use it to read live scoring config. |
| `mygpt-vs` | `vs_endpoint` | Vector Search endpoint (STANDARD) | bundle (`prevent_destroy`) | Hosts the corpus vector index. |
| `projects/mygpt` (display name "MyGPT app state") | `pg_project`, `pg_display_name` | Lakebase Postgres project (PG 17, 0.5–4 CU autoscaling) | bundle | App state: libraries, the per-user upload index, history and feedback. The project creates the `production` branch, its `primary` read-write endpoint and the `databricks_postgres` database. |
| `projects/mygpt/branches/production` | `pg_branch` (app) | Lakebase branch | job (created with the project) | The branch the app connects to (`CAN_CONNECT_AND_CREATE`). `mygpt_upload_ingest` writes through `…/endpoints/primary`. |
| `mygpt` | `secret_scope` | Secret scope | bundle (`prevent_destroy`) | Holds the gateway and tracing credentials (see §8). You put the values in by hand. |

## 2. Unity Catalog: catalog, schemas, volumes

| Name / ID | Variable | Type | Owner | Role |
|---|---|---|---|---|
| `structural_biology` | `catalog` | UC catalog | pre-existing (admin) | Holds every MyGPT schema. |
| `structural_biology.mygpt` | `schema` | UC schema | pre-existing (`deploy.sh` step 1) | Corpus tables, vector index, volumes, UC models, eval datasets and inference tables. |
| `structural_biology.mygpt_tools` | `tools_schema` | UC schema | job (`mygpt_deploy_tools`) | The UC functions that managed MCP serves to the Supervisor as tools. |
| `structural_biology.mygpt.docs` | `docs_volume` | UC volume (managed) | bundle (`prevent_destroy`) | The governed corpus PDFs, one top-level folder per corpus (`biomed`, `health_policies_fr/es/hi`, `kinase_literature`). The folder name is the agent's `custom_inputs.corpus`. |
| `structural_biology.mygpt.uploads` | `uploads_volume` | UC volume | bundle | User upload libraries (`uploads/<library_id>/<doc_id>__<file>`). The app writes here (`WRITE_VOLUME`) and `mygpt_upload_ingest` reads from here. |
| `structural_biology.mygpt.reference` | `reference_volume` | UC volume | bundle | MyGPT-evaluations CSVs: the kinome/kinase seed plus eval questions and gold (`make reference-csvs`). |
| `structural_biology.mygpt.eval_private` | `eval_private_volume` | UC volume (restricted, no grants) | bundle | St. Jude's **private** ground truths (`make eval-private`). Only the eval jobs read it. |

## 3. Tables and views (`structural_biology.mygpt`)

### Corpus ETL (job `mygpt_etl`)

| Name | Type | Built by | Role |
|---|---|---|---|
| `documents` | FILE EXTERNAL table (Beta) | `01_landing_documents.sql` | One governed pointer per file in the `docs` volume, with its corpus, size and mtime. Rebuilt every run. |
| `_docs_in_scope`, `_docs_to_parse` | Delta (staging) | `02_parse_chunk_index.sql` | Working sets for incremental parsing (new or changed files only). |
| `doc_parse_log` | Delta | `02_parse_chunk_index.sql` | The parsed version of each document. Delete a row to force a re-parse. |
| `_doc_elements_stage` | Delta (staging) | `02` + `parse_fallback.py` | Freshly parsed `ai_parse_document` elements (or PyMuPDF elements for flagged docs). |
| `doc_parse_quality` | Delta | `parse_fallback.py` | Parse-quality flags for garbled or script-less documents (e.g. Hindi without Devanagari). |
| `doc_elements` | Delta | `02b_chunk_merge.sql` | Section-tagged parsed elements for every in-scope document. |
| `_doc_chunks_stage` | Delta (staging) | `02b_chunk_merge.sql` | New chunk set before the MERGE. |
| `doc_chunks` | Delta (CDF on) | `02b_chunk_merge.sql` | **Source table of the vector index.** Holds about 1000-character section-bounded chunks with a stable `chunk_id`, plus `corpus` and `section`. |
| `paper_sections` | Delta | `02` / `02b` | Section catalogue per document. The app reads it (`SELECT`) for `/api/sections`, and the agent uses it for section filters. |
| `dataset_config` | Delta | `03_sections_and_config.sql` | Per-corpus / per-library scoring coefficients and retrieval / translation settings (from `etl/config/dataset_config.csv`). Includes `default_<language>` rows for upload libraries. |
| `embedding_model_cutoffs` | Delta | `03_sections_and_config.sql` | Per-embedding-model distance cutoffs. |
| `reranker_config` | Delta | `03_sections_and_config.sql` | Reranker thresholds (English `qnli-electra-base`, multilingual `mmarco-mMiniLMv2`). |
| `kinome` | Delta | `structured_seed.py` | Human kinome synonyms, group and family. A Genie table. |
| `kinase_papers` | Delta | `structured_seed.py` | PubMed ↔ kinase pairs. A Genie table. |
| `corpus_documents` | View | `structured_seed.py` | Every governed document with its corpus and chunk count (`documents` ⟕ `doc_chunks`). A Genie table. |

### Vector index

| Name | Variable | Type | Owner | Role |
|---|---|---|---|---|
| `structural_biology.mygpt.doc_chunks_corpus_index` | `vs_index` | Vector Search Delta Sync index (TRIGGERED, managed embeddings via `mygpt-bge-m3`) | job (`mygpt_etl` → `sync_index`; created on first run) | Corpus-aware hybrid retrieval for the RAG agent, filtered on `corpus` and `section`. Sync pipeline `0ad9f206-0079-4c94-b57c-4310b7c5c79e`. |
| `structural_biology.mygpt.doc_chunks_index` | — (hardcoded, Phase 1) | Vector Search Delta Sync index | pre-existing (Phase 1, superseded) | The Phase-1 index, before corpus filtering. It is still on `mygpt-vs` with sync pipeline `a574e68a-cbaa-4f0c-8015-12f212d5e9de`, and nothing in the current code queries it. Candidate for deletion. |

### Eval datasets (job `mygpt_eval_build_datasets`)

UC-backed MLflow eval datasets. **They contain St. Jude gold in `expectations`, so keep `SELECT` limited to the project team.**

| Name | Role |
|---|---|
| `eval_health_policies_fr` | 21 questions × 10 French health policies |
| `eval_health_policies_es` | 21 questions × 10 Spanish health policies |
| `eval_health_policies_hi` | 21 questions × the public Hindi policies |
| `eval_kinase_literature` | 3 questions × 94 paper/kinase pairs |
| `eval_bioasq` | 163 BioASQ questions with public gold |

### AI Gateway inference tables

`mygpt_deploy_chat_gateway` and `mygpt_deploy_translate_gateway` create these when they enable AI Gateway.

| Name | Role |
|---|---|
| `structural_biology.mygpt.mygpt_chat_payload` | Request/response log of the `mygpt-chat` gateway |
| `structural_biology.mygpt.mygpt_translate_payload` | Request/response log of the `mygpt-translate` gateway |
| `structural_biology.mygpt.mygpt_rag_agent_payload` | Request/response log of `mygpt-rag-agent`. `agents.deploy` creates it; the cost dashboard counts requests from it. |
| `structural_biology.mygpt.mygpt_supervisor_payload` | Request/response log of `mygpt-supervisor`, created the same way. Its successful requests are the app's grounded answers, used for the cost dashboard's $ per answer. |

### System tables read

| Name | Role |
|---|---|
| `system.mlflow.runs_latest`, `system.mlflow.experiments_latest` | The source data for the benchmarks dashboard (eval run metrics, read directly with no ETL). |
| `system.billing.usage`, `system.billing.list_prices` | Cost dashboard: every billed DBU/DSU in the workspace and its list price. |
| `system.serving.endpoint_usage`, `system.serving.served_entities` | Cost dashboard: requests and tokens per endpoint and caller, used for FMAPI token-share allocation. |
| `system.query.history` | Cost dashboard: MyGPT's share of the SQL warehouse, and Genie queries. |
| `system.lakeflow.jobs`, `system.lakeflow.job_run_timeline` | Cost dashboard: MyGPT's jobs, run-hours, and eval-run windows (judge attribution). |
| `system.access.audit` | Cost dashboard: which service principals query MyGPT's vector indexes (the RAG agent's serving identities). |

### Lakebase (Postgres) tables: `databricks_postgres.mygpt`

The app's service principal creates and owns the `mygpt` schema (`pg_schema`) on first start (`app/server/db/migrations.ts`). Row-level security is applied per user, and the `vector` extension is enabled.

| Name | Role |
|---|---|
| `libraries`, `library_documents` | User upload libraries and their documents (owner and sharing). |
| `ingest_runs` | Status of each `mygpt_upload_ingest` run per library. |
| `upload_chunks` | Chunks and bge-m3 embeddings of uploaded documents (pgvector + full text) for upload-library search. |
| `conversations`, `questions`, `answers`, `sources` | Chat history and the cited sources. |
| `feedback` | User ratings. These are also written to MLflow as `user_rating` assessments on the agent traces. |

## 4. UC functions, connections, Genie

| Name / ID | Type | Owner | Role |
|---|---|---|---|
| `rcsb_search` | UC HTTP connection (`https://search.rcsb.org/rcsbsearch/v2`) | job (`mygpt_deploy_tools`) | Governed egress to RCSB PDB search. |
| `rcsb_data` | UC HTTP connection (`https://data.rcsb.org/`) | job (`mygpt_deploy_tools`) | Governed egress to RCSB PDB entry data. |
| `ncbi_eutils` | UC HTTP connection (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils`) | job (`mygpt_deploy_tools`) | Governed egress to the NCBI E-utilities (PubMed). |
| `mygpt_tools.pubmed_ids_from_text` | UC function | job (`mygpt_deploy_tools`) | Extracts PMIDs from text. |
| `mygpt_tools.pubmed_summary` | UC function (uses `ncbi_eutils`) | job (`mygpt_deploy_tools`) | Returns PubMed metadata (title, journal, date, authors, DOI) for a list of PMIDs. |
| `mygpt_tools.search_pdb_by_gene` | UC function (uses `rcsb_search` / `rcsb_data`) | job (`mygpt_deploy_tools`) | Returns the best-resolved PDB structures for a gene. |
| `mygpt_tools.search_pdb` | UC function (uses `rcsb_search` / `rcsb_data`) | job (`mygpt_deploy_tools`) | Full-text search of the PDB. |
| `mygpt_tools.get_pdb_details` | UC function (uses `rcsb_data`) | job (`mygpt_deploy_tools`) | Returns details of one PDB entry. |
| Managed MCP over `structural_biology.mygpt_tools` | Databricks managed MCP server | Databricks (built in) | Exposes the functions above to the Supervisor as tools. |
| "MyGPT structured data" (`genie_title`; ID `01f1c8e7192f12ee9aa687a1ff463507`) | Genie space | job (`mygpt_deploy_genie`; found by title) | Natural-language SQL over `kinome`, `kinase_papers` and `corpus_documents`. The Supervisor routes structured-data questions here. |

## 5. Model Serving and AI Gateway endpoints

### Custom / agent endpoints

| Name | Variable | Type | Owner | Role |
|---|---|---|---|---|
| `mygpt-bge-m3` | `embedding_endpoint` | Custom Model Serving endpoint (GPU_SMALL, Small, no scale-to-zero, AI Gateway usage tracking) serving UC model `bge_m3` | job (`mygpt_deploy_embedding`) | BAAI bge-m3 multilingual embeddings. It is the Vector Search index's managed-embedding model and the upstream of `mygpt-embeddings`. |
| `mygpt-rag-agent` | `rag_agent_endpoint` | Agent (ResponsesAgent) serving endpoint, scale-to-zero, serving UC model `mygpt_rag_agent` | job (`mygpt_deploy_agent`) | The MyGPT RAG pipeline. It retrieves from the corpus index, reranks, generates (Sonnet / Llama / gpt-oss), translates when needed, and scores the answer (ARS, NLI, random-forest hallucination model). Rerankers ship inside the model. This endpoint is the eval target and is called by the Supervisor. |
| `mygpt-supervisor` | `supervisor_endpoint` | Agent (ResponsesAgent) serving endpoint, scale-to-zero, serving UC model `mygpt_supervisor` | job (`mygpt_deploy_supervisor`) | **The app's grounded backend.** It routes each question to the RAG agent (documents), Genie (structured data) or the MCP tools (PubMed / PDB), using `mygpt-chat` as its LLM. |

### Unity AI Gateway endpoints

| Name | Variable | Upstream | Type | Owner | Role |
|---|---|---|---|---|---|
| `mygpt-embeddings` | `embeddings_gateway` | `mygpt-bge-m3` | AI Gateway (`llm/v1/embeddings`, external-model proxy) with usage tracking | job (`mygpt_deploy_gateway`) | Governed embeddings. The RAG agent uses it to embed queries, `mygpt_upload_ingest` to embed upload chunks, and the app to embed upload-library queries. Vector Search can't use it, so the index calls `mygpt-bge-m3` directly. |
| `mygpt-chat` | `chat_gateway` | `databricks-claude-sonnet-5` (`chat_upstream`). An optional Azure AI Foundry fallback route exists. | AI Gateway (`llm/v1/chat`) with usage tracking, an inference table and a per-user rate limit | job (`mygpt_deploy_chat_gateway`) | The agents' default answer model and the Supervisor's LLM. |
| `mygpt-translate` | `translate_gateway` | `databricks-gemma-3-12b` (`translate_upstream`) | AI Gateway (`llm/v1/chat`) with usage tracking, an inference table and a per-user rate limit | job (`mygpt_deploy_translate_gateway`) | Translates queries and answers for non-English corpora (FR / ES / HI). |

### Foundation Model API (pay-per-token, Databricks-hosted)

| Name | Variable | Used by | Role |
|---|---|---|---|
| `databricks-claude-sonnet-5` | `fm_sonnet`, `chat_upstream` | App (direct chat, default), RAG agent, `mygpt-chat` | Default generation model ("sonnet" in the eval matrix). |
| `databricks-llama-4-maverick` | `fm_llama` | App (direct chat), RAG agent | An alternative generation model ("llama" in the eval matrix). |
| `databricks-gpt-oss-120b` | `fm_gptoss` | App (direct chat), RAG agent | An alternative generation model ("gptoss" in the eval matrix). |
| `databricks-gemma-3-12b` | `translate_upstream` | `mygpt-translate` | Translation model. |
| `databricks-claude-opus-5-5` | `judge_model` | Eval and monitoring jobs | The LLM judge for every scorer. It must outrank the agents' models (`eval/scorers.check_judge_outranks`). |

## 6. UC registered models (`structural_biology.mygpt`)

| Name | Variable | Registered by | Role |
|---|---|---|---|
| `bge_m3` | `embedding_model` | `mygpt_deploy_embedding` | The bge-m3 sentence-transformers model behind `mygpt-bge-m3`. |
| `mygpt_rag_agent` | `rag_agent_model` | `mygpt_deploy_agent` | The RAG ResponsesAgent, with its scoring modules and random-forest artifact. Served by `mygpt-rag-agent`. |
| `mygpt_supervisor` | `supervisor_model` | `mygpt_deploy_supervisor` | The Supervisor ResponsesAgent. Served by `mygpt-supervisor`. |

## 7. Jobs (platform bundle)

Every job runs on serverless compute (environment client `1`). Display names start with `name_prefix` (`[mygpt]`).

| Bundle key | Job ID | Role |
|---|---|---|
| `mygpt_etl` | `761945995429746` | Corpus ETL: land → parse (`ai_parse_document`) → PyMuPDF fallback → section-chunk and MERGE → scoring config. It then syncs and creates the vector index and builds the Genie seed tables. |
| `mygpt_upload_ingest` | `352016610803853` | The app triggers it per `library_id` to parse, chunk and embed one upload library into Lakebase `upload_chunks`. Up to 4 runs at once. |
| `mygpt_deploy_embedding` | `503574317781501` | Registers `bge_m3` and creates or updates `mygpt-bge-m3`. |
| `mygpt_deploy_gateway` | `24777479123034` | Creates or updates the `mygpt-embeddings` AI Gateway endpoint. |
| `mygpt_deploy_chat_gateway` | `417654277862604` | Creates or updates `mygpt-chat` and its AI Gateway features. |
| `mygpt_deploy_translate_gateway` | `466125158322504` | Creates or updates `mygpt-translate` and its AI Gateway features. |
| `mygpt_deploy_tools` | `395733236938013` | Creates the HTTP connections and the `mygpt_tools` UC functions. |
| `mygpt_deploy_genie` | `1017939745452066` | Creates or updates the "MyGPT structured data" Genie space. |
| `mygpt_deploy_agent` | `980370336706025` | Logs and registers `mygpt_rag_agent` and deploys `mygpt-rag-agent`. |
| `mygpt_deploy_supervisor` | `805583889716948` | Logs and registers `mygpt_supervisor` and deploys `mygpt-supervisor`. |
| `mygpt_eval_build_datasets` | `577968562754939` | Builds and refreshes the `eval_*` UC eval datasets from the reference and private volumes. |
| `mygpt_eval` | `84251565370622` | `mlflow.genai.evaluate` over a dataset × model matrix against `mygpt-rag-agent`, with optional quality gates (`eval/gates.yaml`). |
| `mygpt_eval_reference` | `274637418766234` | MyGPT parity baseline and judge-agreement scores from MyGPT's recorded answers. It doesn't call any endpoint. |
| `mygpt_eval_monitor` | `1097559855392953` | Registers the production-monitoring scorers (sampled) on the shared experiment. |
| — (MLflow-managed) | `488240277362123` | **"[2750091371639406] Trace Metrics Computation Job"**: MLflow creates this job when the monitoring scorers are registered. It runs them over the production traces on a schedule. It isn't a bundle resource, but it is MyGPT's largest job cost. |

## 8. Secrets (scope `mygpt`)

Only the key names are in config. You put the values in by hand with `databricks secrets put-secret mygpt <key>`.

| Key | Variable | Role |
|---|---|---|
| `gw_upstream_token` | `gw_token_secret_key` | Token with `CAN QUERY` on the gateways' upstream endpoints. It is required by all three gateways. |
| `azure_foundry_key` | `azure_key_secret_key` | Optional Azure AI Foundry API key for the `mygpt-chat` fallback route. |
| `langsmith_api_key` | `langsmith_api_key_secret_key` | Optional. If this key exists, the agents mirror their traces to LangSmith. |

## 9. MLflow, dashboard, app

| Name / ID | Variable | Type | Owner | Role |
|---|---|---|---|---|
| `/Shared/mygpt/mygpt-rag-agent` (ID `2750091371639406`) | `experiment_path`, `experiment_id` (app) | MLflow experiment | bundle (the `/Shared/mygpt` folder must already exist) | One shared experiment for the agent lifecycle: LoggedModels, offline eval runs, production traces of both agent endpoints, monitoring scorers and user-feedback assessments. The app has `CAN_EDIT`. |
| "[mygpt] Benchmarks — MyGPT vs Databricks agent" (ID `01f1c89fd2dc1a41944bc76a54244c5f`) | `dashboard_display_name` | AI/BI (Lakeview) dashboard, `embed_credentials: true` | bundle | Eval scorecard, trend, robustness, kinase configs and scorer trust, read from `system.mlflow`. Generated by `dashboards/build_dashboard.py`. |
| "[mygpt] Costs — usage and spend per component" | `cost_dashboard_display_name` | AI/BI (Lakeview) dashboard, `embed_credentials: true` | bundle | What every component above costs per day: native usage, DBUs and list $. Shared resources are attributed (see `dashboards/README.md`). Generated by `dashboards/build_cost_dashboard.py`. |
| `st-jude-mygpt` | `app_name` | Databricks App (AppKit: Node / React), OBO scope `serving.serving-endpoints` | app bundle | The St. Jude MyGPT chat UI. It does direct chat through FMAPI, grounded chat through `mygpt-supervisor`, upload libraries (volume + ingest job + Lakebase), history and feedback. |

### App resource bindings (`app/databricks.yml` → `app/app.yaml` env)

| App resource | Bound to | Permission | Env var |
|---|---|---|---|
| `serving-endpoint` | `databricks-claude-sonnet-5` | CAN_QUERY | `DATABRICKS_SERVING_ENDPOINT_NAME` |
| `serving-endpoint-llama` | `databricks-llama-4-maverick` | CAN_QUERY | `DATABRICKS_SERVING_ENDPOINT_LLAMA` |
| `serving-endpoint-gptoss` | `databricks-gpt-oss-120b` | CAN_QUERY | `DATABRICKS_SERVING_ENDPOINT_GPTOSS` |
| `agent-endpoint` | `mygpt-supervisor` | CAN_QUERY | `MYGPT_AGENT_ENDPOINT` |
| `embeddings-endpoint` | `mygpt-embeddings` | CAN_QUERY | `MYGPT_EMBED_ENDPOINT` |
| `postgres` | `projects/mygpt/branches/production`, database `databricks-postgres` | CAN_CONNECT_AND_CREATE | `LAKEBASE_ENDPOINT` (+ `PG*`) |
| `uploads-volume` | `structural_biology.mygpt.uploads` | WRITE_VOLUME | `MYGPT_UPLOAD_VOLUME` |
| `paper-sections` | `structural_biology.mygpt.paper_sections` | SELECT | — |
| `upload-job` | job `352016610803853` (`mygpt_upload_ingest`) | CAN_MANAGE_RUN | `MYGPT_UPLOAD_JOB_ID` |
| `sql-warehouse` | `79388385e899d08b` | CAN_USE | `MYGPT_WAREHOUSE_ID` |
| `experiment` | `2750091371639406` | CAN_EDIT | `MYGPT_EXPERIMENT_ID` |

## External (non-Databricks) services

These aren't Databricks resources, but the system calls them:
- **LangSmith:** an optional trace mirror. It uses `langsmith_project` and `langsmith_endpoint`; the current deployment overrides both to point at St. Jude's self-hosted instance.
- **RCSB PDB and NCBI E-utilities:** reached only through the UC HTTP connections in §4.
- **Azure AI Foundry:** an optional chat fallback route, not configured yet.
