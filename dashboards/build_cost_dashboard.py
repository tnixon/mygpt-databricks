"""Generate mygpt_costs.lvdash.json: what MyGPT costs, per component, from the system tables.

    python3 dashboards/build_cost_dashboard.py --profile mygpt   # resolve workspace IDs, then build
    python3 dashboards/build_cost_dashboard.py                   # rebuild from dashboards/cost_ids.json

Resource names come from the platform bundle's variables, read the way the bundle reads them:
databricks.yml defaults, overlaid by .databricks/bundle/default/variable-overrides.json (or the
example file in a fresh checkout); the app name comes from the app bundle's overrides. A few IDs
aren't derivable from names in SQL (the Genie space, the Vector Search sync pipelines, the
Lakebase project uid, the app's service principal, the dashboards); `--profile` resolves them with
read-only API calls and records them in dashboards/cost_ids.json, so later builds run offline.
Re-run after renaming a resource, recreating the index or the Genie space, then redeploy.

Every dataset is sql/costs/base.sql (the attribution CTEs) + sql/costs/<name>.sql. Don't
hand-edit the generated JSON. The attribution rules are explained in dashboards/README.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import lvdash
from lvdash import (agg, cat, chart, col, compact_fmt, counter, daily, date_range_filter, filter_widget,
                    measure, measure_field, num, parameter_widget, pct_fmt, pie, pivot, table, temporal, text,
                    usd_fmt)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SQL = os.path.join(HERE, "sql", "costs")
OUT = os.path.join(HERE, "mygpt_costs.lvdash.json")
IDS = os.path.join(HERE, "cost_ids.json")
LOOKBACK_DAYS = 180  # how far back every dataset reads; the date filter narrows it

DATASETS = {
    "cost_daily": "MyGPT cost per day x component",
    "fm_tokens": "FMAPI tokens by model and attribution",
    "serving_hourly": "Custom serving endpoints: DBUs per hour",
    "requests_daily": "Requests per endpoint and day",
    "unit_economics": "Daily cost vs traffic served",
    "job_runs": "MyGPT job runs",
    "warehouse_share": "SQL warehouse: MyGPT vs other query time",
    "reconciliation": "Workspace bill: MyGPT vs other workloads",
}
PARAMETERS = {"cost_daily": [("discount_pct", "Discount off list (%)", "DECIMAL", "0")]}
MEASURES = {
    "unit_economics": [
        measure("Runtime $ per grounded answer", "SUM(`runtime_usd`) / NULLIF(SUM(`answers`), 0)",
                "Runtime list $ divided by successful mygpt-supervisor requests (the app's grounded answers)"),
        measure("Runtime $ per RAG request", "SUM(`runtime_usd`) / NULLIF(SUM(`rag_requests`), 0)",
                "Runtime list $ divided by successful mygpt-rag-agent requests (app + eval traffic)"),
    ],
    "fm_tokens": [
        measure("MyGPT token share", "SUM(CASE WHEN `owner` = 'MyGPT' THEN `tokens` ELSE 0 END) / NULLIF(SUM(`tokens`), 0)",
                "MyGPT's tokens as a fraction of all tokens on the FMAPI endpoints it uses"),
    ],
}

# Category colors, pinned so a category looks the same on every chart (Okabe-Ito, colorblind-safe).
CATEGORY_COLORS = {
    "Serving endpoints": "#0072B2", "Foundation models": "#E69F00", "Vector Search": "#009E73",
    "Lakebase": "#CC79A7", "App": "#56B4E9", "Jobs": "#D55E00", "SQL & Genie": "#F0E442", "Evaluation": "#999999",
}
OWNER_COLORS = {"MyGPT": "#0072B2", "Other workloads": "#BBBBBB"}
THEME = {
    "canvasBackgroundColor": {"light": "#FCFCFC", "dark": "#1F272D"},
    "widgetBackgroundColor": {"light": "#FFFFFF", "dark": "#11171C"},
    "fontColor": {"light": "#11171C", "dark": "#E8ECF0"},
    "selectionColor": {"light": "#2272B4", "dark": "#8ACAFF"},
    "visualizationColors": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#999999"],
    "widgetHeaderAlignment": "LEFT",
}


# ------------------------------------------------------------------ configuration
def yaml_defaults(path: str) -> tuple[dict[str, str], str]:
    """`variables:` defaults and the default target's host from a bundle file (flat YAML only)."""
    defaults: dict[str, str] = {}
    host = ""
    section, var = None, None
    for line in open(path):
        if re.match(r"^\S", line):
            section = line.split(":")[0].strip()
            continue
        if section == "variables":
            m = re.match(r"^  ([A-Za-z_]\w*):\s*$", line)
            if m:
                var = m.group(1)
            m = re.match(r"^    default:\s*(.*?)\s*$", line)
            if m and var:
                defaults[var] = m.group(1).strip().strip('"').strip("'")
        if section == "targets":
            m = re.match(r"^\s+host:\s*(\S+)", line)
            if m and not host:
                host = m.group(1)
    return defaults, host


def bundle_vars(bundle_dir: str) -> tuple[dict[str, str], str]:
    values, host = yaml_defaults(os.path.join(bundle_dir, "databricks.yml"))
    deployed = os.path.join(bundle_dir, ".databricks", "bundle", "default", "variable-overrides.json")
    example = os.path.join(bundle_dir, "variable-overrides.example.json")
    for path in (deployed, example):
        if os.path.exists(path):
            with open(path) as f:
                values.update({k: str(v) for k, v in json.load(f).items()})
            break
    return values, host


def resolve_ids(profile: str, v: dict[str, str], app_name: str) -> dict:
    """Read-only lookups of the IDs the SQL needs but can't derive from resource names."""
    from databricks.sdk import WorkspaceClient

    sys.path.insert(0, os.path.join(ROOT, "agents", "genie"))
    from driver import find_space  # agents/genie/driver.py: the same title lookup the Supervisor uses

    w = WorkspaceClient(profile=profile)
    pipelines = []
    for ix in w.vector_search_indexes.list_indexes(endpoint_name=v["vs_endpoint"]):
        spec = w.vector_search_indexes.get_index(ix.name).delta_sync_index_spec
        if spec and spec.pipeline_id:
            pipelines.append(spec.pipeline_id)
    app = w.apps.get(app_name)
    titles = {f"{v['name_prefix']} {v['dashboard_display_name']}": "benchmarks",
              f"{v['name_prefix']} {v['cost_dashboard_display_name']}": "costs"}
    dashboards = {titles[d.display_name]: d.dashboard_id for d in w.lakeview.list() if d.display_name in titles}
    return {
        "genie_space_id": find_space(w, v["genie_title"]) or "",
        "vs_pipeline_ids": sorted(set(pipelines)),
        "pg_project_uid": w.postgres.get_project(f"projects/{v['pg_project']}").uid,
        "app_id": app.id,
        "app_sp_client_id": app.service_principal_client_id,
        "experiment_id": w.experiments.get_by_name(v["experiment_path"]).experiment.experiment_id,
        "dashboard_ids": dashboards,
        # Ids of MyGPT callers that the SQL can't infer (e.g. an agent identity that never called a
        # MyGPT gateway). Add by hand; kept across re-resolves.
        "extra_requesters": [],
    }


def sql_list(items: list[str], empty: str = "''") -> str:
    return ", ".join(f"'{i}'" for i in items) if items else empty


def placeholders(v: dict[str, str], host: str, app_name: str, ids: dict) -> dict[str, str]:
    m = re.search(r"adb-(\d+)\.", host)
    if not m:
        raise SystemExit(f"can't read the workspace id from host {host!r}")
    fq = f"{v['catalog']}.{v['schema']}"
    fm = [v[k] for k in ("fm_sonnet", "fm_llama", "fm_gptoss", "judge_model", "chat_upstream", "translate_upstream")]
    requesters = [ids["app_id"], ids["app_sp_client_id"], *ids.get("extra_requesters", [])]
    return {
        "workspace_id": m.group(1), "lookback_days": str(LOOKBACK_DAYS), "name_prefix": v["name_prefix"],
        "experiment_id": ids["experiment_id"], "catalog": v["catalog"], "schema": v["schema"],
        "rag_agent_endpoint": v["rag_agent_endpoint"], "supervisor_endpoint": v["supervisor_endpoint"],
        "embedding_endpoint": v["embedding_endpoint"], "embeddings_gateway": v["embeddings_gateway"],
        "chat_gateway": v["chat_gateway"], "chat_upstream": v["chat_upstream"],
        "translate_gateway": v["translate_gateway"], "translate_upstream": v["translate_upstream"],
        "judge_model": v["judge_model"], "vs_endpoint": v["vs_endpoint"], "warehouse_id": v["warehouse_id"],
        "app_name": app_name, "genie_space_id": ids["genie_space_id"], "pg_project_uid": ids["pg_project_uid"],
        "fm_endpoints": ", ".join(f"('{e}')" for e in dict.fromkeys(fm)),
        "mygpt_requesters": ", ".join(f"('{r}')" for r in dict.fromkeys(filter(None, requesters))),
        "vs_pipeline_ids": sql_list(ids["vs_pipeline_ids"]),
        "dashboard_ids": sql_list(sorted(ids["dashboard_ids"].values())),
        # agents.deploy names the inference tables after the endpoint
        "supervisor_payload": f"{fq}.{v['supervisor_endpoint'].replace('-', '_')}_payload",
        "rag_agent_payload": f"{fq}.{v['rag_agent_endpoint'].replace('-', '_')}_payload",
    }


def fill(sql: str, values: dict[str, str]) -> str:
    for k, val in values.items():
        sql = sql.replace("{" + k + "}", val)
    left = sorted(set(re.findall(r"\{([a-z_]+)\}", sql)))
    if left:
        raise SystemExit(f"unfilled placeholders: {left}")
    return sql


def dataset(name: str, values: dict[str, str]) -> dict:
    with open(os.path.join(SQL, "base.sql")) as f:
        base = f.read().rstrip("\n")
    with open(os.path.join(SQL, f"{name}.sql")) as f:
        body = f.read().rstrip("\n")
    return lvdash.dataset(name, DATASETS[name], fill(base + "\n" + body, values),
                          PARAMETERS.get(name), MEASURES.get(name))


# ------------------------------------------------------------------ layout helpers
def colored(field: str, label: str, colors: dict[str, str]) -> dict:
    enc = cat(field, label)
    enc["scale"]["mappings"] = [{"value": k, "color": c} for k, c in colors.items()]
    return enc


def only(column: str, *values: str) -> list[str]:
    return [f"`{column}` IN ({sql_list(list(values))})"]


def stacked_daily(name: str, ds: str, title: str, value: str, label: str, color: str, color_label: str,
                  pos: tuple[int, int, int, int], kind: str = "bar", colors: dict | None = None, fmt: dict | None = None,
                  filters: list[str] | None = None) -> dict:
    x, y = daily("usage_date"), agg("sum", value)
    color_enc = colored(color, color_label, colors) if colors else cat(color, color_label)
    return chart(name, kind, ds, title, temporal(x[0], "Day"), num(y[0], label, None, None, fmt), color_enc, pos,
                 fields=[x, y, (color, f"`{color}`")], filters=filters)


def hbar(name: str, ds: str, title: str, value: str, label: str, dim: str, dim_label: str,
         pos: tuple[int, int, int, int], fmt: dict | None = None, filters: list[str] | None = None,
         color: tuple[str, str, dict] | None = None) -> dict:
    v = agg("sum", value)
    y = cat(dim, dim_label)
    y["scale"]["sort"] = {"by": "x-reversed"}
    color_enc = colored(*color) if color else None
    fields = [v, (dim, f"`{dim}`")] + ([(color[0], f"`{color[0]}`")] if color else [])
    return chart(name, "bar", ds, title, num(v[0], label, None, None, fmt), y, color_enc, pos, fields=fields, filters=filters)


USD = usd_fmt(2)
COMPACT = compact_fmt(1)

HEADER = """# What MyGPT costs
Every Databricks resource in **INVENTORY.md**, read from the system tables (`system.billing`, `system.serving`, `system.query`, `system.lakeflow`) and filtered to MyGPT's own components; the rest of this shared workspace is left out (see *Attribution*).
**List $** = DBUs (or DSUs) × the SKU's effective list price at the time of use (`system.billing.list_prices`); **net $** applies the *Discount off list* filter. Each component also shows its usage in its own unit: endpoint-hours, tokens, requests, run-hours, query-hours."""

HOW_ATTRIBUTED = """### How costs are attributed
- **direct**: billed to a MyGPT resource by name/id (serving endpoints, gateways, `mygpt-vs` and its index-sync pipelines, the Lakebase project, the app, the Genie space, `[mygpt]` jobs and the MLflow trace-metrics job).
- **token share**: the pay-per-token FMAPI endpoints are shared with other workloads. Each day's bill is split by MyGPT's share of the tokens, output tokens weighted 5× input for Claude and 3× for the others.
- **query share**: the SQL warehouse's daily bill × MyGPT's share of its query task time.
- **time window**: MLflow judge (`AGENT_EVALUATION`) usage that falls inside a MyGPT eval / monitoring job run, by the same identity.
Lifecycle: **runtime** = serving the app, **ingest** = ETL / upload / index sync, **eval** = evaluation & monitoring, **deploy** = deploy jobs."""

SERVING_MD = """### Models and serving
The three custom endpoints are billed per serving window whether or not they get traffic: a flat DBU/hour line is an endpoint that stays up. `mygpt-bge-m3` never scales to zero on purpose (Vector Search embeds queries through it); the agents are configured to scale to zero, so a flat line there means traffic (or monitoring) keeps them up.
FMAPI rows count **only MyGPT's tokens**; the *Attribution* page shows its share of each model."""

PLATFORM_MD = """### Data platform
Vector Search bills its endpoint by the hour and its index syncs as serverless pipelines. Lakebase bills compute (scale-to-zero) in DBUs and branch storage in DSUs. The warehouse is shared, so its cost follows MyGPT's share of query time (lower chart)."""

ATTRIBUTION_MD = """### Attribution and reconciliation
The top chart splits the **whole workspace bill** into what this dashboard attributes to MyGPT and everything else (other demos, notebooks, networking, predictive optimization). The token chart shows, per FMAPI model, which rule claimed MyGPT's tokens:
- **gateway**: calls through `mygpt-chat` / `mygpt-translate` / `mygpt-embeddings`, counted on the gateway because the model sees them as the gateway token's owner.
- **mygpt identity**: direct calls by MyGPT's service principals (the agents' serving identities, the app).
- **eval window**: calls by a MyGPT eval / monitoring job's run-as identity during that run (the judges).
- **usage_context**: calls tagged `usage_context.project = mygpt` (the app's direct chat and the agents, once deployed with it)."""

GAPS_MD = """### Known gaps
- **Prices are list prices**: negotiated discounts aren't in the system tables. Use *Discount off list* to approximate them.
- **Unattributed FMAPI calls**: direct calls by an identity that never called a MyGPT gateway and sent no `usage_context` count as *other workloads*. That includes the agents' direct calls before the gateways existed (before 2026-10-15) and the app's direct chat before it sent `usage_context`. Add known ids to `extra_requesters` in `dashboards/cost_ids.json`.
- **Not counted**: Azure storage for the UC tables and volumes (billed by Azure, not Databricks), plus networking and predictive optimization, which carry no resource id.
- **Freshness**: billing and serving usage land a few hours late, so the latest day is partial."""


def build(values: dict[str, str]) -> dict:
    fm_rows = only("category", "Foundation models")
    overview = [
        text("o_header", HEADER, 0, 0, 12, 3),
        counter("o_list", "cost_daily", "list_usd", "List $", 0, 3, 3, 3, USD, period=daily("usage_date")),
        counter("o_net", "cost_daily", "net_usd", "Net $ (after discount)", 3, 3, 3, 3, USD, period=daily("usage_date")),
        counter("o_dbus", "cost_daily", "dbus", "DBUs", 6, 3, 3, 3, COMPACT, period=daily("usage_date")),
        counter("o_runtime", "cost_daily", "list_usd", "Runtime list $", 9, 3, 3, 3, USD, period=daily("usage_date"),
                filters=only("lifecycle", "runtime")),
        stacked_daily("o_daily", "cost_daily", "List $ per day, by category", "list_usd", "List $", "category", "Category",
                      (0, 6, 8, 7), kind="area", colors=CATEGORY_COLORS, fmt=USD),
        pie("o_lifecycle", "cost_daily", "List $ by lifecycle", agg("sum", "list_usd"), "lifecycle", (8, 6, 4, 7), USD),
        hbar("o_components", "cost_daily", "List $ by component", "list_usd", "List $", "component", "Component",
             (0, 13, 6, 9), USD, color=("category", "Category", CATEGORY_COLORS)),
        stacked_daily("o_dbus_daily", "cost_daily", "DBUs per day, by category", "dbus", "DBUs", "category", "Category",
                      (6, 13, 6, 9), colors=CATEGORY_COLORS, fmt=COMPACT),
        pivot("o_summary", "cost_daily", "Every component: usage in its own unit, DBUs and $",
              ["category", "component", "lifecycle", "allocation", "native_unit"],
              [(agg("sum", "native_qty"), "Usage (native unit)", compact_fmt(2)), (agg("sum", "dbus"), "DBUs", COMPACT),
               (agg("sum", "list_usd"), "List $", USD), (agg("sum", "net_usd"), "Net $", USD)],
              (0, 22, 12, 10)),
        text("o_how", HOW_ATTRIBUTED, 0, 32, 12, 5),
    ]
    serving = [
        text("s_header", SERVING_MD, 0, 0, 12, 3),
        counter("s_endpoints", "cost_daily", "list_usd", "Serving endpoints list $", 0, 3, 3, 3, USD,
                period=daily("usage_date"), filters=only("category", "Serving endpoints")),
        counter("s_fm", "cost_daily", "list_usd", "Foundation models list $ (MyGPT share)", 3, 3, 3, 3, USD,
                period=daily("usage_date"), filters=fm_rows),
        counter("s_tokens", "fm_tokens", "tokens", "MyGPT FMAPI tokens", 6, 3, 3, 3, COMPACT,
                period=daily("usage_date"), filters=only("owner", "MyGPT")),
        counter("s_per_answer", "unit_economics", "", "Runtime $ per grounded answer", 9, 3, 3, 3, usd_fmt(3),
                field=measure_field("Runtime $ per grounded answer")),
        chart("s_hourly", "line", "serving_hourly", "DBUs per hour, custom serving endpoints",
              temporal("hour", "Hour"), num("sum(dbus)", "DBUs / hour", None, None, COMPACT), cat("endpoint", "Endpoint"),
              (0, 6, 12, 7), fields=[("hour", "`hour`"), agg("sum", "dbus"), ("endpoint", "`endpoint`")]),
        stacked_daily("s_tokens_daily", "fm_tokens", "MyGPT tokens per day, by model", "tokens", "Tokens", "model", "Model",
                      (0, 13, 6, 7), fmt=COMPACT, filters=only("owner", "MyGPT")),
        stacked_daily("s_fm_daily", "cost_daily", "Foundation-model list $ per day (MyGPT share)", "list_usd", "List $",
                      "component", "Model / gateway", (6, 13, 6, 7), fmt=USD, filters=fm_rows),
        stacked_daily("s_requests", "requests_daily", "Requests per day, by endpoint", "requests", "Requests", "endpoint",
                      "Endpoint", (0, 20, 6, 7), fmt=COMPACT),
        chart("s_unit", "line", "unit_economics", "Runtime $ per grounded answer, by day",
              temporal("daily(usage_date)", "Day"), num("measure(Runtime $ per grounded answer)", "$ per answer", None, None,
                                                        usd_fmt(3)), None,
              (6, 20, 6, 7), fields=[daily("usage_date"), measure_field("Runtime $ per grounded answer")]),
        pivot("s_tokens_pivot", "fm_tokens", "MyGPT tokens by model, input vs output", ["model", "token_type", "attribution"],
              [(agg("sum", "tokens"), "Tokens", compact_fmt(2))], (0, 27, 6, 8), filters=only("owner", "MyGPT")),
        pivot("s_endpoints_pivot", "cost_daily", "Serving & gateway usage",
              ["component", "native_unit"],
              [(agg("sum", "native_qty"), "Usage (native unit)", compact_fmt(2)), (agg("sum", "dbus"), "DBUs", COMPACT),
               (agg("sum", "list_usd"), "List $", USD)],
              (6, 27, 6, 8), filters=only("category", "Serving endpoints", "Foundation models")),
    ]
    platform_cats = ("Vector Search", "Lakebase", "App", "SQL & Genie", "Jobs", "Evaluation")
    platform = [
        text("p_header", PLATFORM_MD, 0, 0, 12, 2),
        counter("p_vs", "cost_daily", "list_usd", "Vector Search list $", 0, 2, 3, 3, USD, period=daily("usage_date"),
                filters=only("category", "Vector Search")),
        counter("p_lakebase", "cost_daily", "list_usd", "Lakebase list $", 3, 2, 3, 3, USD, period=daily("usage_date"),
                filters=only("category", "Lakebase")),
        counter("p_app", "cost_daily", "list_usd", "App list $", 6, 2, 3, 3, USD, period=daily("usage_date"),
                filters=only("category", "App")),
        counter("p_jobs", "cost_daily", "list_usd", "Jobs list $", 9, 2, 3, 3, USD, period=daily("usage_date"),
                filters=only("category", "Jobs")),
        stacked_daily("p_vs_daily", "cost_daily", "Vector Search list $ per day", "list_usd", "List $", "component",
                      "Component", (0, 5, 6, 6), fmt=USD, filters=only("category", "Vector Search")),
        stacked_daily("p_lb_daily", "cost_daily", "Lakebase list $ per day: compute vs storage", "list_usd", "List $",
                      "component", "Component", (6, 5, 6, 6), fmt=USD, filters=only("category", "Lakebase")),
        stacked_daily("p_jobs_daily", "cost_daily", "Jobs list $ per day, by lifecycle", "list_usd", "List $", "lifecycle",
                      "Lifecycle", (0, 11, 6, 6), fmt=USD, filters=only("category", "Jobs")),
        stacked_daily("p_wh_share", "warehouse_share", "SQL warehouse query-hours: MyGPT vs other", "query_hours",
                      "Query-hours", "owner", "Owner", (6, 11, 6, 6), colors=OWNER_COLORS, fmt=compact_fmt(2)),
        pivot("p_summary", "cost_daily", "Data-platform components: usage in its own unit, DBUs and $",
              ["category", "component", "lifecycle", "native_unit"],
              [(agg("sum", "native_qty"), "Usage (native unit)", compact_fmt(2)), (agg("sum", "dbus"), "DBUs", COMPACT),
               (agg("sum", "list_usd"), "List $", USD)],
              (0, 17, 12, 8), filters=only("category", *platform_cats)),
        table("p_runs", "job_runs", "Job runs", [
            col("started", "Started"),
            col("job", "Job"),
            col("lifecycle", "Lifecycle"),
            col("result_state", "Result"),
            col("run_hours", "Run-hours", 2),
            col("dbus", "DBUs", 2),
            col("list_usd", "List $", fmt=USD),
            col("run_id", "Run id"),
        ], (0, 25, 12, 8)),
    ]
    attribution = [
        text("a_header", ATTRIBUTION_MD, 0, 0, 12, 4),
        stacked_daily("a_recon", "reconciliation", "Workspace list $ per day: MyGPT vs other workloads", "list_usd", "List $",
                      "owner", "Owner", (0, 4, 8, 7), kind="area", colors=OWNER_COLORS, fmt=USD),
        counter("a_share", "fm_tokens", "", "MyGPT share of FMAPI tokens (models it uses)", 8, 4, 4, 7, pct_fmt(1),
                field=measure_field("MyGPT token share")),
        hbar("a_tokens", "fm_tokens", "FMAPI tokens by model and attribution rule", "tokens", "Tokens", "model", "Model",
             (0, 11, 12, 7), COMPACT,
             color=("attribution", "Attributed by", {"gateway": "#0072B2", "mygpt identity": "#56B4E9",
                                                     "eval window": "#009E73", "usage_context": "#CC79A7",
                                                     "other workloads": "#BBBBBB"})),
        text("a_gaps", GAPS_MD, 0, 18, 12, 5),
    ]
    all_ds = list(DATASETS)
    filters_page = [
        date_range_filter("g_dates", "usage_date", "Date range", all_ds, (0, 0, 1, 2)),
        filter_widget("g_category", "category", "Category", ["cost_daily"], (0, 2, 1, 2), multi=True),
        filter_widget("g_lifecycle", "lifecycle", "Lifecycle", ["cost_daily"], (0, 4, 1, 2), multi=True),
        parameter_widget("g_discount", "discount_pct", "Discount off list (%)", "cost_daily", (0, 6, 1, 2)),
    ]

    def page(name: str, display: str, layout: list[dict]) -> dict:
        return {"name": name, "displayName": display, "pageType": "PAGE_TYPE_CANVAS", "layoutVersion": "GRID_V1",
                "layout": layout}

    return {
        "datasets": [dataset(n, values) for n in DATASETS],
        "pages": [
            page("overview", "Overview", overview),
            page("serving", "Models & serving", serving),
            page("platform", "Data platform", platform),
            page("attribution", "Attribution", attribution),
            {"name": "global_filters", "displayName": "Filters", "pageType": "PAGE_TYPE_GLOBAL_FILTERS",
             "layout": filters_page},
        ],
        "uiSettings": {"theme": THEME},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", help="resolve workspace IDs with this CLI profile (read-only) and update cost_ids.json")
    ap.add_argument("--print-sql", metavar="DATASET", help="print one dataset's filled SQL (for testing) and exit")
    args = ap.parse_args()

    values, host = bundle_vars(ROOT)
    app_values, _ = bundle_vars(os.path.join(ROOT, "app"))
    app_name = app_values["app_name"]

    ids = json.load(open(IDS)) if os.path.exists(IDS) else None
    if args.profile:
        resolved = resolve_ids(args.profile, values, app_name)
        resolved["extra_requesters"] = (ids or {}).get("extra_requesters", [])
        ids = resolved
        with open(IDS, "w") as f:
            json.dump(ids, f, indent=2)
            f.write("\n")
        print(f"wrote {IDS}", file=sys.stderr)
    if ids is None:
        raise SystemExit(f"no {os.path.relpath(IDS, ROOT)}: run once with --profile <profile>")

    filled = placeholders(values, host, app_name, ids)
    if args.print_sql:
        print("".join(dataset(args.print_sql, filled)["queryLines"]))
        return
    with open(OUT, "w") as f:
        json.dump(build(filled), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
