# MyGPT on Databricks — fresh-workspace deployment guide

This guide deploys the whole MyGPT POC into a **new, empty workspace**, in order. Each step
names the bundle (or manual mechanism) that runs it and the resources it creates.

There are two Databricks Asset Bundles:

| Bundle | Location | Owns |
|---|---|---|
| **platform** (`st-jude-mygpt-platform`) | repo root `databricks.yml` | Lakebase, volumes, experiment, dashboard, and every job (ETL, model/agent deploys, eval) |
| **app** (`st-jude-mygpt`) | `app/databricks.yml` | The Databricks App and its resource bindings |

Most model-serving, gateway and Genie objects are not bundle resources. A platform-bundle
**job** creates each of them when it runs (`bundle run <job>`). That is why the order matters.

---

## Required components and services (checklist)

Before you start, confirm that the target workspace has every capability below. The **Step**
column points to the deployment step that first needs it. Step 13 is optional.

> **⚠️ Preview / Beta features to enable first.** A workspace admin enables these on the
> workspace **Previews** page (Settings → Previews), unless noted otherwise:
> - **File type** (Beta). **Required.** The ETL's `FILE EXTERNAL` / `FILE MANAGED` columns depend on it
>   (`etl/sql/01_landing_documents.sql`). Needs Databricks SQL / DBR 18 LTS or later.
> - **Managed MCP Servers** (Beta). **Required** (step 11). The Supervisor calls the UC-functions and
>   Genie MCP endpoints (`/api/2.0/mcp/functions/…`, `/api/2.0/mcp/genie/…`).
> - **MLflow production monitoring** (Beta). Step 13 only. Registered scorers sample live
>   traces (`eval/monitor.py`).
> - **Metastore-level UC HTTP connections** (Public Preview). **Required** (step 11). `rcsb_search`,
>   `rcsb_data` and `ncbi_eutils` are created at metastore level
>   (`agents/tools/sql/01_connections.sql`). The SQL `http_request()` function the tools use is
>   **deprecated** but still works.
> - **Apps `user_api_scopes`** (on-behalf-of-user auth). OBO went GA on 2026-10-07, but the API
>   field is still labelled Public Preview. If the app deploy rejects `user_api_scopes`, enable
>   user authorization for Apps on the Previews page.
>
> Release statuses were checked against the Databricks docs on 2026-10-18. They change often, so
> re-check the Previews page on the target workspace.

### Platform foundations
| ✓ | Capability | Used for | Step |
|---|---|---|---|
| ☐ | Unity Catalog metastore, with permission to create catalog `structural_biology` (or an existing one) | Every table, volume, model, function and index | 1 |
| ☐ | Serverless jobs compute (environment client `1`) | All 14 platform-bundle jobs | 2 |
| ☐ | Serverless SQL warehouse | ETL SQL, the app's section catalogue, Genie | 1 |
| ☐ | Secret scopes | `mygpt/gw_upstream_token` (and optional `azure_foundry_key`) | 1, 4 |
| ☐ | Databricks Asset Bundles + CLI with OAuth | Both bundles | 2 |
| ☐ | A service principal | Owns the gateway upstream token (CAN QUERY on its upstream endpoints) | 4 |

### Data and document processing
| ✓ | Capability | Used for | Step | Status |
|---|---|---|---|---|
| ☐ | UC volumes | `docs`, `uploads`, `eval_private` | 1, 2 | GA |
| ☐ | **File type** (`FILE EXTERNAL` / `FILE MANAGED`) | `documents` landing table | 7 | ⚠️ **Beta: enable on Previews** |
| ☐ | `ai_parse_document` | PDF parsing in the ETL and `mygpt_upload_ingest` | 7 | GA (needs serverless env 3+ / DBR 17.3+) |
| ☐ | Delta Change Data Feed | `doc_chunks` → Delta Sync index | 7 | GA |

### Retrieval
| ✓ | Capability | Used for | Step | Status |
|---|---|---|---|---|
| ☐ | Vector Search (AI Search) STANDARD endpoint | `mygpt-vs` | 8 | GA |
| ☐ | Delta Sync index with managed embeddings on a **custom model** endpoint | `doc_chunks_corpus_index` embedding through `mygpt-bge-m3` | 8 | GA |

AI Search won't accept an external-model (gateway) endpoint as the managed embedding model.
That's why the index points at `mygpt-bge-m3` directly, not `mygpt-embeddings`.

### Model Serving and agents
| ✓ | Capability | Used for | Step | Status |
|---|---|---|---|---|
| ☐ | **GPU** Model Serving (GPU_SMALL, always-on) | `mygpt-bge-m3`. GPU capacity/quota must exist in the region | 3 | GA |
| ☐ | CPU custom-model serving + Mosaic AI Agent Framework (`agents.deploy`, `ResponsesAgent`) | `mygpt-rag-agent`, `mygpt-supervisor` | 10, 11 | GA |
| ☐ | Foundation Model API pay-per-token: `databricks-claude-sonnet-5`, `databricks-llama-4-maverick`, `databricks-gpt-oss-120b`, `databricks-gemma-3-12b` | Chat, translation, app direct chat | 5, 10, 12 | GA. Availability varies by cloud/region, so check each one in the Serving UI |
| ☐ | AI Gateway on serving endpoints: external models (`databricks-model-serving` provider), usage tracking, inference tables, rate limits, fallback | `mygpt-embeddings`, `mygpt-chat`, `mygpt-translate` | 5 | GA (Unity Gateway GA since 2026-08-04) |
| ☐ | *(optional)* Azure AI Foundry deployment (OpenAI provider) | Fallback served entity on `mygpt-chat` | 5 | n/a |

### Application and state
| ✓ | Capability | Used for | Step | Status |
|---|---|---|---|---|
| ☐ | Databricks Apps | `st-jude-mygpt` | 12 | GA |
| ☐ | Apps on-behalf-of-user auth (`user_api_scopes: serving.serving-endpoints`) | The app calls serving endpoints as the signed-in user | 12 | GA. ⚠️ the `user_api_scopes` field is still labelled Public Preview |
| ☐ | Lakebase (autoscaling projects, PG 17) with the `vector` (pgvector) extension | App state, upload-library embeddings | 2, 12 | GA (Azure since 2026-03-02) |

### MLflow and evaluation
| ✓ | Capability | Used for | Step | Status |
|---|---|---|---|---|
| ☐ | MLflow 3 experiments and tracing | `/Shared/mygpt/mygpt-rag-agent`, which holds models, traces and feedback | 2, 10 | GA |
| ☐ | AI/BI dashboards | Benchmarks dashboard | 2 | GA |
| ☐ | MLflow GenAI evaluation + UC eval datasets | `mygpt_eval*` jobs | 13 | GA |
| ☐ | MLflow **production monitoring** (registered scorers) | `mygpt_eval_monitor` | 13 | ⚠️ **Beta** |

### Supervisor, tools and Genie (step 11)
| ✓ | Capability | Used for | Status |
|---|---|---|---|
| ☐ | UC HTTP connections (metastore level) + `CREATE CONNECTION` privilege | `rcsb_search`, `rcsb_data`, `ncbi_eutils` | ⚠️ **Public Preview** (schema-level connections are GA) |
| ☐ | UC SQL functions calling `http_request()` | PubMed / RCSB tools in `structural_biology.mygpt_tools` | ⚠️ `http_request()` is **deprecated** |
| ☐ | **Managed MCP servers** (UC functions, Genie) | How the Supervisor calls tools and Genie | ⚠️ **Beta: enable on Previews** |
| ☐ | Genie spaces + Genie space management API | "MyGPT structured data" space | GA |
| ☐ | Outbound internet from serverless to `search.rcsb.org`, `data.rcsb.org`, `eutils.ncbi.nlm.nih.gov` | Tool calls | Check serverless egress / network policies |

---

## 0. Before you start

### Tools
- Databricks CLI with an OAuth profile for the target workspace:
  `databricks auth login --host https://<workspace-url> --profile <profile>`
- `git`, `make`, and Python with `databricks-sdk` (for the local smoke tests).
- Clone **with submodules**. The Genie seed tables and the eval datasets are read from
  `reference/MyGPT-evaluations`:
  ```bash
  git clone --recurse-submodules <repo-url>
  ```

### Workspace capabilities
Work through the [checklist](#required-components-and-services-checklist) above, including the
preview/beta toggles. Note the **serverless SQL warehouse** id; you'll need it as `$WH`.

### Configuration (bundle variables)
Every resource name, display name and infrastructure id is a bundle variable: catalog / schema,
volumes, endpoint and UC model names, the Genie title, the experiment path, the Lakebase project,
job and dashboard name prefixes, and secret scope / key **names** (never secret values). Each
variable's default in `databricks.yml` is the current St. Jude deployment value.

Each bundle reads its values from `.databricks/bundle/default/variable-overrides.json`, a
gitignored file kept per checkout. Install both files from the committed examples, then edit them:
```bash
make config     # copies variable-overrides.example.json -> .databricks/bundle/default/ (root + app/); never overwrites
```
| File (committed example) | Bundle | Keys |
|---|---|---|
| `variable-overrides.example.json` | platform | all platform variables (see `variables:` in `databricks.yml` for descriptions) |
| `app/variable-overrides.example.json` | app | `app_name`, `catalog`, `schema`, `uploads_volume`, FMAPI / Supervisor / embeddings endpoint names, `warehouse_id`, `upload_job_id`, `experiment_id`, `pg_branch` |

Precedence: `--var name=value` > env `BUNDLE_VAR_<name>` > overrides file > `databricks.yml`
default. Check the resolved values with `databricks bundle validate -o json`.

**Keep these keys in step across the two files:**
| Platform key | App key |
|---|---|
| `catalog`, `schema`, `uploads_volume` | same names |
| `supervisor_endpoint`, `embeddings_gateway` | same names |
| `fm_sonnet`, `fm_llama`, `fm_gptoss` | same names |
| `pg_project` | `pg_branch` = `projects/<pg_project>/branches/production` |
| `experiment_path`, job `mygpt_upload_ingest` (both created in step 2) | `experiment_id`, `upload_job_id` (read from `databricks bundle summary`) |

Notes:
- The ETL and tools SQL take `catalog` / `schema` / `docs_volume` / `tools_schema` as job
  parameters. The app derives its schema from the `uploads-volume` resource. Changing the catalog
  therefore only means editing the overrides files (and `CATALOG=` / `SCHEMA=` for the `make`
  upload targets).
- The dashboard JSON is static. After changing `experiment_path`, run
  `python3 dashboards/build_dashboard.py --experiment <path>`, then redeploy.
- Changing `app_name` creates a **new** app.
- Not variables: the workspace host (`targets.default.workspace.host` in **both** bundles: set
  it to the new workspace URL), the UC HTTP connection names (`rcsb_search`, `rcsb_data`,
  `ncbi_eutils`, used as literals inside the tool functions), the Lakebase Postgres schema the app
  creates (`MYGPT_PG_SCHEMA` in `app/app.yaml`; keep it equal to `pg_schema`), and the
  `etl/config/*.csv` model ids.
- CLI profile: the `Makefile` default is `stjude-mygpt`. Pass `PROFILE=<profile>` to every `make` target.

The commands below use these placeholders:
```bash
P=<profile>            # CLI profile
WH=<warehouse_id>      # serverless SQL warehouse id (goes in the overrides files as warehouse_id)
```
Run every platform-bundle command from the **repo root**, and every app-bundle command from `app/`.

> **Rule:** `bundle run` executes the **last-deployed** copy of the code. After changing anything,
> run `bundle deploy` again before `bundle run`. This applies to both bundles.

---

## Deployment steps at a glance

| # | Step | Bundle / mechanism | Creates |
|---|---|---|---|
| 1 | Bootstrap UC, workspace folder, secret scope | manual | schema `mygpt`, volume `docs`, `/Shared/mygpt`, secret scope `mygpt` |
| 2 | Deploy the platform bundle | platform `bundle deploy` | Lakebase project, `uploads` + `eval_private` volumes, experiment, dashboard, 14 jobs |
| 3 | Deploy the embedding model | platform `run mygpt_deploy_embedding` | UC model `bge_m3`, endpoint `mygpt-bge-m3` |
| 4 | Store the gateway upstream token | manual | secret `mygpt/gw_upstream_token` |
| 5 | Deploy the AI Gateway endpoints | platform `run mygpt_deploy_gateway`, `…_chat_gateway`, `…_translate_gateway` | `mygpt-embeddings`, `mygpt-chat`, `mygpt-translate` |
| 6 | Upload the document corpora | `make` / `databricks fs cp` | PDFs in the `docs` volume |
| 7 | First ETL run | platform `run mygpt_etl` | Document, chunk, config, section and Genie seed tables |
| 8 | Create the Vector Search endpoint + index | manual (one-time) | VS endpoint `mygpt-vs`, index `doc_chunks_corpus_index` |
| 9 | Second ETL run | platform `run mygpt_etl` | Index synced and row-count checked |
| 10 | Deploy the RAG agent | platform `run mygpt_deploy_agent` | UC model `mygpt_rag_agent`, endpoint `mygpt-rag-agent` |
| 11 | Supervisor, tools, Genie | platform `run mygpt_deploy_tools`, `…_genie`, `…_supervisor` | UC connections + functions, Genie space, endpoint `mygpt-supervisor` |
| 12 | Deploy the app | app `bundle deploy` + `bundle run app` | App `st-jude-mygpt`; Lakebase schema `mygpt` on first start |
| 13 | *(optional)* Evaluation + monitoring | `make eval-private` + platform `run mygpt_eval_*` | UC eval datasets, eval runs, monitoring scorers |

---

## 1. Bootstrap Unity Catalog, workspace folder and secret scope
**Mechanism:** manual, run once.

The platform bundle puts volumes in `structural_biology.mygpt` and an experiment under
`/Shared/mygpt`, so both must exist before step 2. The corpora (step 6) are uploaded to the
`docs` volume before the ETL runs for the first time. The ETL's landing SQL would create these
too, but only after it is too late.

Run in the SQL editor (on `$WH`):
```sql
CREATE CATALOG IF NOT EXISTS structural_biology;   -- or have an admin create it with a managed location
CREATE SCHEMA  IF NOT EXISTS structural_biology.mygpt
  COMMENT 'MyGPT POC — document corpus, indexes, sessions';
CREATE VOLUME  IF NOT EXISTS structural_biology.mygpt.docs
  COMMENT 'Raw source documents (FILE EXTERNAL corpus)';
```
Then run:
```bash
databricks workspace mkdirs /Shared/mygpt -p $P
databricks secrets create-scope mygpt -p $P
```

**Creates:** catalog `structural_biology` (if absent), schema `structural_biology.mygpt`, volume
`structural_biology.mygpt.docs`, workspace folder `/Shared/mygpt`, secret scope `mygpt`.

## 2. Deploy the platform bundle
**Bundle:** platform (`bundle deploy`)

Run `make config` first and set `warehouse_id` (plus any names you change) in
`.databricks/bundle/default/variable-overrides.json` (see [Configuration](#configuration-bundle-variables)).
```bash
databricks bundle validate -p $P
databricks bundle deploy   -p $P
```

**Creates** (default names shown):
| Resource | Name |
|---|---|
| Lakebase project | `mygpt` (PG 17, 0.5–4 CU autoscaling), with its `production` branch, `primary` read-write endpoint and `databricks_postgres` database |
| UC volume | `structural_biology.mygpt.uploads` (user upload libraries) |
| UC volume | `structural_biology.mygpt.eval_private` (restricted: owner and metastore admins only) |
| MLflow experiment | `/Shared/mygpt/mygpt-rag-agent` (shared by agent models, eval runs and production traces) |
| AI/BI dashboard | `[mygpt] Benchmarks — MyGPT vs Databricks agent` |
| Jobs (defined, not run) | `mygpt_etl`, `mygpt_upload_ingest`, `mygpt_deploy_embedding`, `mygpt_deploy_gateway`, `mygpt_deploy_chat_gateway`, `mygpt_deploy_translate_gateway`, `mygpt_deploy_agent`, `mygpt_deploy_tools`, `mygpt_deploy_genie`, `mygpt_deploy_supervisor`, `mygpt_eval_build_datasets`, `mygpt_eval`, `mygpt_eval_reference`, `mygpt_eval_monitor` |
| Workspace files | The repo (`etl/`, `agents/`, `eval/`, `dashboards/`, eval CSVs) synced to the bundle's `${workspace.file_path}` |

**Record for step 12:**
```bash
databricks bundle summary -p $P                                          # jobs.mygpt_upload_ingest → job id
databricks experiments get-by-name /Shared/mygpt/mygpt-rag-agent -p $P   # experiment_id
```

## 3. Deploy the bge-m3 embedding model
**Bundle:** platform (`bundle run mygpt_deploy_embedding`) → `agents/embedding/driver.py`

```bash
databricks bundle run mygpt_deploy_embedding -p $P
```

**Creates:**
- UC registered model `structural_biology.mygpt.bge_m3` (BAAI/bge-m3, MLflow pyfunc).
- Model Serving endpoint **`mygpt-bge-m3`**: GPU_SMALL, **always-on** (scale-to-zero off).
  Vector Search embeds every query through it, so a cold endpoint would make queries fail.

**Verify:** wait until the endpoint is READY (Serving UI, or
`databricks serving-endpoints get mygpt-bge-m3 -p $P`). Optionally run
`python agents/embedding/smoke_test.py --profile $P`.

## 4. Store the gateway upstream token
**Mechanism:** manual.

The AI Gateway endpoints (`databricks-model-serving` provider) authenticate to their upstream
endpoints with a Databricks token. Use a **service-principal** token, not a personal PAT, with
CAN QUERY on `mygpt-bge-m3`, `databricks-claude-sonnet-5` and `databricks-gemma-3-12b`.
```bash
databricks secrets put-secret mygpt gw_upstream_token -p $P
# optional, only if routing chat to Azure AI Foundry as a fallback:
databricks secrets put-secret mygpt azure_foundry_key -p $P
```

**Creates:** secret `mygpt/gw_upstream_token` (plus `mygpt/azure_foundry_key` if used).

## 5. Deploy the Unity AI Gateway endpoints
**Bundle:** platform (three jobs, all running `agents/gateway/driver.py --task …`)

```bash
databricks bundle run mygpt_deploy_gateway           -p $P
databricks bundle run mygpt_deploy_chat_gateway      -p $P
databricks bundle run mygpt_deploy_translate_gateway -p $P
```

**Creates:**
| Job | Endpoint | Routes to | Gateway features |
|---|---|---|---|
| `mygpt_deploy_gateway` | `mygpt-embeddings` | `mygpt-bge-m3` (`llm/v1/embeddings`) | Routing only |
| `mygpt_deploy_chat_gateway` | `mygpt-chat` | `databricks-claude-sonnet-5` (`llm/v1/chat`) | Usage tracking, inference table `structural_biology.mygpt.mygpt_chat_*`, 60 calls/min/user rate limit, optional Azure AI Foundry fallback |
| `mygpt_deploy_translate_gateway` | `mygpt-translate` | `databricks-gemma-3-12b` (`llm/v1/chat`) | Same as `mygpt-chat` (inference table `mygpt_translate_*`) |

To add the Azure AI Foundry fallback to `mygpt-chat`, add
`--azure-endpoint https://<res>.openai.azure.com --azure-deployment <name>` to the job's
parameters in `databricks.yml`, then deploy and re-run.

**Verify:**
```bash
python agents/gateway/smoke_test.py --profile $P
python agents/gateway/smoke_test.py --profile $P --task chat
python agents/gateway/smoke_test.py --profile $P --task translate
```

## 6. Upload the document corpora
**Mechanism:** `make` targets or `databricks fs cp` (not a bundle).

Each top-level folder in the `docs` volume is one **corpus**. The agent filters retrieval on it
(`custom_inputs.corpus`).
```bash
make eval-corpus PROFILE=$P [KINASE_PDFS=<dir of <pubmed_id>.pdf>]   # biomed, health_policies_fr/_es, kinase_literature
make corpus-hi   PROFILE=$P HI_PDFS=<dir>                             # health_policies_hi (public Hindi PDFs)
```
The ETL picks up only the corpora on the allow-list in `etl/sql/02_parse_chunk_index.sql`. To
add a new corpus, follow the checklist in [`etl/README.md`](etl/README.md#today-the-mygpt_etl-job-incremental).

**Creates:** PDFs under `/Volumes/structural_biology/mygpt/docs/<corpus>/`.

## 7. First ETL run
**Bundle:** platform (`bundle run mygpt_etl`)

```bash
databricks bundle run mygpt_etl -p $P
```

**Creates** (all in `structural_biology.mygpt`):
| Task | Tables / views |
|---|---|
| `landing` | `documents` (one pointer row per file) |
| `parse_chunk_index` | `doc_elements`, `doc_chunks` (Change Data Feed on), `doc_parse_log`, `doc_parse_quality` |
| `parse_fallback` | Re-extracts garbled or script-less documents with PyMuPDF and records the result in `doc_parse_quality` |
| `chunk_merge` | MERGEs section chunks into `doc_chunks`; builds `paper_sections` |
| `sections_and_config` | `dataset_config`, `embedding_model_cutoffs`, `reranker_config` |
| `structured_seed` | Genie seed tables `kinome`, `kinase_papers`, view `corpus_documents` |
| `sync_index` | *(see below)* |

> **Expected on the first run:** the **`sync_index` task fails** because the Vector Search index
> doesn't exist yet. The other tasks still complete. Step 8 creates the index, and step 9
> re-runs the job.

## 8. Create the Vector Search endpoint and index (one-time)
**Mechanism:** manual CLI. Index creation is not part of any bundle or job.

```bash
databricks vector-search-endpoints create-endpoint mygpt-vs STANDARD -p $P
# wait until the endpoint is ONLINE, then:
databricks vector-search-indexes create-index -p $P --json '{
  "name": "structural_biology.mygpt.doc_chunks_corpus_index",
  "endpoint_name": "mygpt-vs",
  "primary_key": "chunk_id",
  "index_type": "DELTA_SYNC",
  "delta_sync_index_spec": {
    "source_table": "structural_biology.mygpt.doc_chunks",
    "pipeline_type": "TRIGGERED",
    "embedding_source_columns": [
      {"name": "chunk_text", "embedding_model_endpoint_name": "mygpt-bge-m3"}
    ],
    "columns_to_sync": ["chunk_id","doc_id","source_name","corpus","section","page","elem_type","chunk_text"]
  }
}'
```
The index embeds through the **model** endpoint `mygpt-bge-m3`, not the `mygpt-embeddings`
gateway: AI Search rejects external-model (gateway) endpoints as managed embedding models.

**Creates:** Vector Search endpoint `mygpt-vs`; Delta Sync index
`structural_biology.mygpt.doc_chunks_corpus_index`. The initial sync starts on creation.

## 9. Second ETL run
**Bundle:** platform (`bundle run mygpt_etl`)

```bash
databricks bundle run mygpt_etl -p $P
```
Unchanged files aren't re-parsed, so this run is quick. `sync_index` now triggers a sync, waits
for it, and fails if the index's `indexed_row_count` doesn't equal `count(doc_chunks)`.

**Verify:** every task is green. To check parse quality:
`SELECT corpus, parser, reason, count(*) FROM structural_biology.mygpt.doc_parse_quality GROUP BY ALL`.

## 10. Deploy the RAG agent
**Bundle:** platform (`bundle run mygpt_deploy_agent`) → `agents/rag_agent/driver.py`

Prerequisites: steps 3, 5 and 8. MLflow runs a full validation `predict` when the model is
logged, and that predict calls the index, the embedding endpoints and the chat/translate
gateways live.
```bash
databricks bundle run mygpt_deploy_agent -p $P
```

**Creates:**
- A LoggedModel in the shared experiment, and the UC registered model
  `structural_biology.mygpt.mygpt_rag_agent`.
- Model Serving endpoint **`mygpt-rag-agent`** (MLflow `ResponsesAgent`, scale-to-zero). Its
  serving identity is granted the declared resources: the index `doc_chunks_corpus_index`,
  `mygpt-embeddings`, `mygpt-bge-m3`, `mygpt-chat`, `mygpt-translate`, and the three FMAPI chat
  endpoints.

**Verify:** the endpoint is READY. A test query in the Serving UI returns an answer with
`custom_outputs` scores (QRS / ARS / HI) and sources. The first calls after a deploy can take
30+ s while the rerankers download.

## 11. Supervisor, domain tools and Genie
**Bundle:** platform (three jobs, in this order)

The app's grounded chat runs through the Supervisor (`mygpt-supervisor`), which routes each
question to the selected libraries' documents (via `mygpt-rag-agent`), Genie or the tools, so
deploy it before the app.
```bash
databricks bundle run mygpt_deploy_tools      -p $P
databricks bundle run mygpt_deploy_genie      -p $P
databricks bundle run mygpt_deploy_supervisor -p $P
```

**Creates:**
| Job | Resources |
|---|---|
| `mygpt_deploy_tools` | UC HTTP connections `rcsb_search`, `rcsb_data`, `ncbi_eutils`. Schema `structural_biology.mygpt_tools` with UC functions (PubMed ids/summaries, RCSB PDB search/details, governed corpora), served via managed MCP |
| `mygpt_deploy_genie` | Genie space **"MyGPT structured data"** over `kinome`, `kinase_papers` and `corpus_documents` (from step 7), on `$WH` |
| `mygpt_deploy_supervisor` | UC model `structural_biology.mygpt.mygpt_supervisor`. Endpoint **`mygpt-supervisor`** (scale-to-zero), with resources on `mygpt-chat`, `mygpt-rag-agent`, the Genie space, its tables, the tool functions and the connections |

**Verify:** `python agents/supervisor/smoke_test.py --profile $P` (documents, genie and tools routes, and a
two-library question that should search only the kinase library).

## 12. Deploy the app
**Bundle:** app (`bundle deploy` + `bundle run app`), run from `app/`.

```bash
cd app
databricks bundle validate -p $P
databricks bundle deploy   -p $P
databricks bundle run app  -p $P
```
Before this, set `warehouse_id`, `upload_job_id` and `experiment_id` in
`app/.databricks/bundle/default/variable-overrides.json` to the values recorded in step 2. On a
fresh workspace, **don't** run the `bundle deployment bind` mentioned in `PLAN.md`. That step only applies to the
original sandbox, where the app already existed.

**Creates:** Databricks App **`st-jude-mygpt`**, which starts on deploy. It calls serving
endpoints on behalf of the signed-in user (OBO scope `serving.serving-endpoints`). Its service
principal is granted:
| App resource | Target | Permission | Created in step |
|---|---|---|---|
| `serving-endpoint`, `-llama`, `-gptoss` | FMAPI Sonnet / Llama / GPT-OSS | CAN_QUERY | (workspace) |
| `agent-endpoint` | `mygpt-supervisor` | CAN_QUERY | 11 |
| `embeddings-endpoint` | `mygpt-embeddings` | CAN_QUERY | 5 |
| `postgres` | Lakebase `projects/mygpt/branches/production` / `databricks-postgres` | CAN_CONNECT_AND_CREATE | 2 |
| `uploads-volume` | `structural_biology.mygpt.uploads` | WRITE_VOLUME | 2 |
| `paper-sections` | `structural_biology.mygpt.paper_sections` | SELECT | 7 |
| `upload-job` | `mygpt_upload_ingest` | CAN_MANAGE_RUN | 2 |
| `sql-warehouse` | `$WH` | CAN_USE | (prereq) |
| `experiment` | `/Shared/mygpt/mygpt-rag-agent` | CAN_EDIT | 2 |

The app calls `mygpt-supervisor` **as the signed-in user**, so app users need CAN_QUERY on it
(Serving UI → `mygpt-supervisor` → Permissions, e.g. for the users group). They need nothing on
`mygpt-rag-agent`, Genie or the tools: the Supervisor calls those with its own credentials.

On first start, the app's service principal creates and owns the Lakebase schema `mygpt` and
runs migrations `001_init` → `005_supervisor`. These cover libraries, upload chunks with
pgvector, RLS, history and feedback, and seed the corpus libraries.

**Verify:** the app logs show `[lakebase] schema mygpt ready`. If they show
`[lakebase] UNAVAILABLE`, the app is running in stateless mode with no history or uploads; check
the Lakebase resource and the `CREATE EXTENSION vector` permission. Then open the app URL and
ask a question against a governed library.

When a user uploads files to a library, the app itself triggers **`mygpt_upload_ingest`**, which
parses, chunks and embeds them into Lakebase `mygpt.upload_chunks`. That job is never run by
hand.

## 13. *(Optional)* Evaluation and production monitoring
**Bundle:** platform

```bash
make eval-private PROFILE=$P                                              # private SME ground truths → eval_private volume
databricks bundle run mygpt_eval_build_datasets -p $P
databricks bundle run mygpt_eval                -p $P  # matrix of dataset × model; gate=false by default
databricks bundle run mygpt_eval_reference      -p $P  # MyGPT parity baseline + judge agreement
databricks bundle run mygpt_eval_monitor        -p $P
```

**Creates:** files in the `eval_private` volume; UC MLflow eval datasets in
`structural_biology.mygpt`; eval runs in `/Shared/mygpt/mygpt-rag-agent` (these feed the
Benchmarks dashboard from step 2); production-monitoring scorers registered on the experiment,
which score a sample of live `mygpt-rag-agent` traces. Without the private ground truths, only
the public BioASQ gold is available. See [`eval/README.md`](eval/README.md).

---

## Operating notes
- **Redeploy before running.** After any code change, run `bundle deploy` for the bundle you
  changed, then `bundle run`.
- **Never `CREATE OR REPLACE` `doc_chunks`.** Replacing it gives the table a new id, which
  breaks the index's change feed (`DIFFERENT_DELTA_TABLE_READ_BY_STREAMING_SOURCE`). The only
  fix is to rebuild the index.
- **If the index is deleted and recreated, redeploy the agent** (step 10). The new index lacks
  the agent's SELECT grant, so every query returns `retrieve_error: PermissionDenied` until then.
- **Force a re-parse** of a document by deleting its `doc_parse_log` row and re-running `mygpt_etl`.
- **Rotate `mygpt/gw_upstream_token`** before it expires, or all three gateway endpoints stop working.

## Teardown
```bash
cd app && databricks bundle destroy -p $P && cd ..
databricks bundle destroy -p $P
```
`bundle destroy` doesn't remove the objects created by jobs or by hand. Delete these separately:
the serving endpoints (`mygpt-bge-m3`, `mygpt-embeddings`, `mygpt-chat`, `mygpt-translate`,
`mygpt-rag-agent`, `mygpt-supervisor`); the VS index and the `mygpt-vs` endpoint; the Genie
space; the UC models, tables, the `docs` volume and the `mygpt` / `mygpt_tools` schemas; the
three HTTP connections; and the `mygpt` secret scope.
