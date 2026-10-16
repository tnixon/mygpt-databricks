"""Log, register, and deploy the MyGPT Supervisor ResponsesAgent (Phase 5).

Run as a DABs job task (`mygpt_deploy_supervisor`) after the chat gateway, the RAG agent,
the tool functions and the Genie space exist. It:
  1. resolves the Genie space id by title (agents/genie/driver.py creates it) and lists the
     UC functions in the tools schema;
  2. logs `supervisor_agent.py` (code-based) with every hop declared as a resource, so Agent
     Framework provisions the endpoint's credentials for them;
  3. registers it to Unity Catalog and deploys it with Mosaic AI Agent Framework.

Declare every hop (Phase-1b lesson): the chat gateway, the RAG endpoint, the Genie space plus
its tables and warehouse, each tool function, the HTTP connections the functions call, and
the tables the corpus functions read. The RAG agent declares its own index / embedding
endpoints on its endpoint.

    python agents/supervisor/driver.py --catalog structural_biology --schema mygpt --warehouse-id <id>
"""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksUCConnection,
)

GENIE_TITLE = "MyGPT structured data"  # = agents/genie/driver.py TITLE
CONNECTIONS = ["rcsb_search", "rcsb_data", "ncbi_eutils"]  # = agents/tools/sql/01_connections.sql
GENIE_TABLES = ["kinome", "kinase_papers", "corpus_documents"]
# corpus_documents is a view over these; the identity running the query needs them too.
VIEW_INPUTS = ["documents", "doc_chunks"]


def _package_dir(base_dir: str) -> str:
    """On a Databricks job `__file__` is undefined, so the job passes --base-dir ${workspace.file_path}."""
    if base_dir:
        return os.path.join(base_dir, "agents", "supervisor")
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.join(os.getcwd(), "agents", "supervisor")


def find_genie_space(w, title: str) -> str:
    token = None
    while True:
        page = w.api_client.do("GET", "/api/2.0/genie/spaces", query={"page_token": token} if token else None)
        for s in page.get("spaces") or []:
            if s.get("title") == title:
                return s["space_id"]
        token = page.get("next_page_token")
        if not token:
            raise SystemExit(f"Genie space {title!r} not found: run mygpt_deploy_genie first")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--tools-schema", default="mygpt_tools")
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--model-name", default="mygpt_supervisor")
    ap.add_argument("--endpoint-name", default="mygpt-supervisor")
    ap.add_argument("--llm-endpoint", default="mygpt-chat")
    ap.add_argument("--rag-endpoint", default="mygpt-rag-agent")
    ap.add_argument("--genie-title", default=GENIE_TITLE)
    # Same shared experiment as the RAG agent: LoggedModels, eval runs and production traces.
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    here = _package_dir(args.base_dir)
    fq = f"{args.catalog}.{args.schema}"
    tools_schema = f"{args.catalog}.{args.tools_schema}"
    uc_model = f"{fq}.{args.model_name}"
    space_id = find_genie_space(w, args.genie_title)
    functions = sorted(f.full_name for f in w.functions.list(catalog_name=args.catalog, schema_name=args.tools_schema))
    if not functions:
        raise SystemExit(f"no functions in {tools_schema}: run mygpt_deploy_tools first")
    print(f"Genie space {space_id}; {len(functions)} tool functions: {', '.join(f.rsplit('.', 1)[-1] for f in functions)}")

    resources = [
        DatabricksServingEndpoint(endpoint_name=args.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=args.rag_endpoint),
        DatabricksGenieSpace(genie_space_id=space_id),
        DatabricksSQLWarehouse(warehouse_id=args.warehouse_id),
        *[DatabricksTable(table_name=f"{fq}.{t}") for t in GENIE_TABLES + VIEW_INPUTS],
        *[DatabricksFunction(function_name=f) for f in functions],
        *[DatabricksUCConnection(connection_name=c) for c in CONNECTIONS],
    ]
    env = {
        "MYGPT_SUPERVISOR_LLM": args.llm_endpoint,
        "MYGPT_RAG_ENDPOINT": args.rag_endpoint,
        "MYGPT_GENIE_SPACE_ID": space_id,
        "MYGPT_TOOLS_SCHEMA": tools_schema,
    }
    # The log-time validation predict imports the agent in this process; give it the same config.
    os.environ.update(env)

    mlflow.set_registry_uri("databricks-uc")
    experiment_id = mlflow.set_experiment(args.experiment).experiment_id
    with mlflow.start_run():
        logged = mlflow.pyfunc.log_model(
            name="agent",
            python_model=os.path.join(here, "supervisor_agent.py"),
            pip_requirements=os.path.join(here, "requirements.txt"),
            resources=resources,
            # A tool question: exercises the LLM + managed-MCP path without a RAG call.
            input_example={"input": [{"role": "user", "content": "Which document collections can I search?"}]},
        )
    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"Registered {uc_model} v{registered.version} (LoggedModel {logged.model_id})")

    from databricks import agents

    agents.deploy(
        uc_model,
        registered.version,
        endpoint_name=args.endpoint_name,
        # Pin traces to the shared experiment (agents.deploy otherwise keeps whatever an
        # existing endpoint had; see the RAG driver).
        environment_vars={**env, "MLFLOW_EXPERIMENT_ID": experiment_id},
        scale_to_zero_enabled=True,
    )
    print(f"Deploy requested for endpoint {args.endpoint_name}")


if __name__ == "__main__":
    main()
