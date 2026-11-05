# Dashboards

Two AI/BI dashboards, deployed with the platform bundle (`resources.dashboards`). Each `*.lvdash.json`
is generated; edit the SQL and the generator, never the JSON.

| Dashboard | Generator | SQL | Reads |
|---|---|---|---|
| **[mygpt] Benchmarks — MyGPT vs Databricks agent** | `build_dashboard.py` | `sql/*.sql` | `system.mlflow` (see eval/README.md) |
| **[mygpt] Costs — usage and spend per component** | `build_cost_dashboard.py` | `sql/costs/*.sql` | `system.billing`, `system.serving`, `system.query`, `system.lakeflow`, `system.access.audit`, the agents' payload tables |

Widget builders shared by both are in `lvdash.py`.

## Cost dashboard

**What it shows.** Every component in INVENTORY.md, per day:
- usage in its native unit: endpoint-hours, tokens, requests, run-hours, query-hours, app-hours, DSU
- DBUs
- list-price $ (`system.billing.list_prices`, effective list price at the time of use)
- net $ after the *Discount off list* filter

**Pages:**
- **Overview**: totals, cost by category, lifecycle and component, and a pivot of every component.
- **Models & serving**: DBU/hour of the custom endpoints, MyGPT's FMAPI tokens and $, requests, runtime $ per grounded answer.
- **Data platform**: Vector Search, Lakebase, the app, the warehouse share, jobs and every job run.
- **Attribution**: the workspace bill split into MyGPT vs other workloads, and which rule claimed each model's tokens.

**Rebuild** after renaming a resource, recreating the vector index or Genie space, or editing the SQL:

```bash
python3 dashboards/build_cost_dashboard.py --profile mygpt   # re-resolve IDs into cost_ids.json, then build
python3 dashboards/build_cost_dashboard.py                   # build from the recorded IDs (offline)
python3 dashboards/build_cost_dashboard.py --print-sql cost_daily   # one dataset's SQL, to test it
databricks bundle deploy -p mygpt
```

Resource names come from the bundle variables (`databricks.yml` + `.databricks/bundle/default/variable-overrides.json`).
`--profile` resolves the IDs that can't be derived from names in SQL, using read-only API calls:
- the Genie space
- the Vector Search sync pipelines
- the Lakebase project uid
- the app and its service principal
- the experiment
- the two dashboards

The first build after the cost dashboard is first deployed should use `--profile`. That records its own id, so its queries count as MyGPT's warehouse time.

### Attribution rules

The workspace is shared with other workloads, so nothing is counted by default: every source is
matched to MyGPT explicitly (`sql/costs/base.sql`).

| Rule | Applies to | How |
|---|---|---|
| **direct** | serving endpoints, AI Gateway fees, `mygpt-vs` and its index-sync pipelines, the Lakebase project, the app, the Genie space, `ai_parse_document` on the MyGPT warehouse, jobs | Billing rows matched by endpoint / app / project / pipeline / space id. A job counts if its name starts with `name_prefix`, plus `[<experiment id>] Trace Metrics Computation Job`, which MLflow creates for the monitoring scorers. |
| **token share** | pay-per-token FMAPI endpoints (`fm_sonnet`, `fm_llama`, `fm_gptoss`, `judge_model`, the gateway upstreams) | Each day's bill (serving + AI Gateway fee) × MyGPT's share of that endpoint's tokens, with output weighted 5× input for Claude and 3× for the others. MyGPT's tokens are claimed by the rules below. |
| **query share** | the SQL warehouse | Each day's bill × MyGPT's share of query task time. MyGPT's queries come from its jobs, its two dashboards, its Genie space, and its service principals or app. |
| **time window** | `AGENT_EVALUATION` (managed judges) | Billing rows whose run-as identity and time window match a MyGPT eval or monitoring job run. |

FMAPI tokens count as MyGPT's when they meet any of these rules:
- **gateway**: the call went through `mygpt-chat`, `mygpt-translate` or `mygpt-embeddings`. These are counted on the gateway, because the model sees gateway traffic as the gateway token's owner. That owner's own rows on the upstream are never claimed.
- **mygpt identity**: the caller is a MyGPT service principal. That means an SP that called a MyGPT endpoint, an SP that queried a `<catalog>.<schema>.*` vector index (the RAG agent's per-version serving identities, from `system.access.audit`), the app, or any id added to `extra_requesters` in `cost_ids.json`.
- **eval window**: the caller is a MyGPT eval or monitoring job's run-as identity, during that run. This covers the judges. It isn't applied to gateway upstreams.
- **usage_context**: the request carried `usage_context.project = "mygpt"`. The agents send it, and the app's direct chat sends it as `component = "app-direct"`.

**Lifecycle:**
- `runtime`: serving the app
- `ingest`: ETL, upload ingest and index sync
- `eval`: evals, monitoring and judges
- `deploy`: the deploy jobs

**Not counted:**
- Azure storage for the UC tables and volumes (billed by Azure).
- Networking and predictive optimization, which have no resource id.
- People's direct FMAPI calls outside MyGPT jobs, unless they're tagged with `usage_context`.

**Prices are list prices.** Use *Discount off list* to approximate a negotiated rate.

**Permissions.** The dashboard embeds the deployer's credentials (`embed_credentials: true`). The deployer needs SELECT on:
- `system.billing`, `system.serving`, `system.query`, `system.lakeflow` and `system.access`
- the agents' payload tables

**Freshness.** Billing and serving usage arrive a few hours late, so the most recent day is partial.
