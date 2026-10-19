"""Deployment helpers behind the Makefile targets (stdlib only; shells out to the databricks CLI).

    python3 scripts/bundle_config.py get <key> [--app]        # platform (or app) value of a variable (Makefile helper)
    python3 scripts/bundle_config.py check                    # shared keys agree across the two bundles
    python3 scripts/bundle_config.py sync-ids  --profile P    # platform -> app overrides (+ deployed ids)
    python3 scripts/bundle_config.py bootstrap --profile P    # DEPLOYMENT.md step 1
    python3 scripts/bundle_config.py vs-index  --profile P    # DEPLOYMENT.md step 8

Values come from each bundle's .databricks/bundle/default/variable-overrides.json (`make config`),
falling back to the variable defaults in variable-overrides.example.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app")
OVERRIDES = os.path.join(".databricks", "bundle", "default", "variable-overrides.json")

# Platform key -> app key, for every value the app bundle repeats.
SHARED = {k: k for k in ("catalog", "schema", "uploads_volume", "fm_sonnet", "fm_llama", "fm_gptoss",
                         "supervisor_endpoint", "embeddings_gateway", "warehouse_id")}
# App keys that must be set before the app deploys (empty defaults in app/databricks.yml).
APP_REQUIRED = ("warehouse_id", "upload_job_id", "experiment_id")


def load(bundle_dir: str) -> dict:
    vals: dict = {}
    for name in ("variable-overrides.example.json", OVERRIDES):
        path = os.path.join(bundle_dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                vals.update(json.load(f))
    return vals


def cli(*args: str, profile: str, cwd: str = ROOT, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["databricks", *args, "-p", profile], cwd=cwd, capture_output=True, text=True, check=check)


def pg_branch(platform: dict) -> str:
    return f"projects/{platform['pg_project']}/branches/production"


def cmd_get(a) -> None:
    print(load(APP if a.app else ROOT).get(a.key, ""))


def cmd_check(_a) -> None:
    platform, app = load(ROOT), load(APP)
    errors = [f"{pk}: platform {platform.get(pk)!r} != app {app.get(ak)!r}"
              for pk, ak in SHARED.items() if platform.get(pk) != app.get(ak)]
    if app.get("pg_branch") != pg_branch(platform):
        errors.append(f"pg_branch: app {app.get('pg_branch')!r} != {pg_branch(platform)!r} (platform pg_project)")
    errors += [f"app {k} is empty (run `make sync-ids` after `make deploy-platform`)" for k in APP_REQUIRED if not app.get(k)]
    if errors:
        sys.exit("config-check failed:\n  " + "\n  ".join(errors))
    print("config-check OK: app overrides agree with the platform bundle")


def cmd_sync_ids(a) -> None:
    platform = load(ROOT)
    summary = json.loads(cli("bundle", "summary", "-o", "json", profile=a.profile).stdout)["resources"]
    job_id = summary["jobs"]["mygpt_upload_ingest"].get("id")
    exp_id = summary["experiments"]["mygpt_rag_agent"].get("id")
    if not job_id or not exp_id:
        sys.exit("platform bundle not deployed yet: run `make deploy-platform` first")

    path = os.path.join(APP, OVERRIDES)
    if not os.path.exists(path):
        sys.exit(f"{path} missing: run `make config` first")
    with open(path, encoding="utf-8") as f:
        app = json.load(f)
    new = {**app, **{ak: platform[pk] for pk, ak in SHARED.items()},
           "pg_branch": pg_branch(platform), "upload_job_id": str(job_id), "experiment_id": str(exp_id)}
    changed = {k: (app.get(k), v) for k, v in new.items() if app.get(k) != v}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
    for k, (old, v) in changed.items():
        print(f"  {k}: {old!r} -> {v!r}")
    print(f"sync-ids: {len(changed)} key(s) updated in {os.path.relpath(path, ROOT)}")


def sql(profile: str, warehouse_id: str, statement: str) -> None:
    body = {"warehouse_id": warehouse_id, "statement": statement, "wait_timeout": "50s"}
    out = json.loads(cli("api", "post", "/api/2.0/sql/statements", "--json", json.dumps(body), profile=profile).stdout)
    state = out["status"]["state"]
    if state != "SUCCEEDED":
        sys.exit(f"{statement}\n  -> {state}: {out['status'].get('error', {}).get('message', '')}")
    print(f"  ok  {statement.split(' COMMENT')[0]}")


def cmd_bootstrap(a) -> None:
    p = load(ROOT)
    if not p.get("warehouse_id"):
        sys.exit("warehouse_id is empty in the platform overrides (make config, then edit it)")
    fq = f"`{p['catalog']}`.`{p['schema']}`"
    sql(a.profile, p["warehouse_id"], f"CREATE SCHEMA IF NOT EXISTS {fq} COMMENT 'MyGPT POC — document corpus, indexes, sessions'")
    sql(a.profile, p["warehouse_id"], f"CREATE VOLUME IF NOT EXISTS {fq}.`{p['docs_volume']}` COMMENT 'Raw source documents (FILE EXTERNAL corpus)'")
    folder = os.path.dirname(p["experiment_path"])
    cli("workspace", "mkdirs", folder, profile=a.profile)
    print(f"  ok  workspace folder {folder}")
    r = cli("secrets", "create-scope", p["secret_scope"], profile=a.profile, check=False)
    if r.returncode and "already exists" not in (r.stderr + r.stdout).lower():
        sys.exit(f"secrets create-scope {p['secret_scope']}: {r.stderr.strip()}")
    print(f"  ok  secret scope {p['secret_scope']}" + (" (already existed)" if r.returncode else ""))
    print(f"bootstrap done. The catalog {p['catalog']!r} must already exist (an admin creates it with a managed location).")


def cmd_vs_index(a) -> None:
    p = load(ROOT)
    endpoint, index = p["vs_endpoint"], f"{p['catalog']}.{p['schema']}.{p['vs_index']}"
    if cli("vector-search-endpoints", "get-endpoint", endpoint, profile=a.profile, check=False).returncode:
        cli("vector-search-endpoints", "create-endpoint", endpoint, "STANDARD", "--no-wait", profile=a.profile)
        print(f"  created VS endpoint {endpoint}")
    while True:
        ep = json.loads(cli("vector-search-endpoints", "get-endpoint", endpoint, "-o", "json", profile=a.profile).stdout)
        state = (ep.get("endpoint_status") or {}).get("state")
        if state == "ONLINE":
            break
        print(f"  VS endpoint {endpoint}: {state}; waiting...")
        time.sleep(30)
    if not cli("vector-search-indexes", "get-index", index, profile=a.profile, check=False).returncode:
        print(f"vs-index: {index} already exists (nothing to do)")
        return
    spec = {
        "name": index, "endpoint_name": endpoint, "primary_key": "chunk_id", "index_type": "DELTA_SYNC",
        "delta_sync_index_spec": {
            "source_table": f"{p['catalog']}.{p['schema']}.doc_chunks",
            "pipeline_type": "TRIGGERED",
            # AI Search rejects gateway endpoints as managed embedding models: use the model endpoint.
            "embedding_source_columns": [{"name": "chunk_text", "embedding_model_endpoint_name": p["embedding_endpoint"]}],
            "columns_to_sync": ["chunk_id", "doc_id", "source_name", "corpus", "section", "page", "elem_type", "chunk_text"],
        },
    }
    cli("vector-search-indexes", "create-index", "--json", json.dumps(spec), profile=a.profile)
    print(f"vs-index: created {index} on {endpoint}; the initial sync has started")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("key")
    g.add_argument("--app", action="store_true", help="read the app bundle instead of the platform bundle")
    sub.add_parser("check")
    for name in ("sync-ids", "bootstrap", "vs-index"):
        sub.add_parser(name).add_argument("--profile", default="mygpt")
    a = ap.parse_args()
    {"get": cmd_get, "check": cmd_check, "sync-ids": cmd_sync_ids,
     "bootstrap": cmd_bootstrap, "vs-index": cmd_vs_index}[a.cmd](a)


if __name__ == "__main__":
    main()
