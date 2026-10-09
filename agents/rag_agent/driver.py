"""Log, register, and deploy the MyGPT ResponsesAgent.

Run on a Databricks cluster/notebook (or as a DABs job task). It:
  1. logs `agent.py` as an MLflow model (code-based logging) with the RF pickle,
     the scoring/config modules, and the declared Databricks resources;
  2. registers it to Unity Catalog;
  3. deploys it to a Model Serving endpoint via Mosaic AI Agent Framework.

The embeddings gateway endpoint (`mygpt-embeddings`, Phase 1b — proxies bge-m3), the
Vector Search index, and the FMAPI chat endpoints are declared as `resources` so Agent
Framework provisions the endpoint's auth to reach them. On-behalf-of-user auth (so AI
Search retrieval runs as the signed-in user) is validated separately — see PLAN.md risks.

Usage:
    python agents/rag_agent/driver.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)


def _package_dir(base_dir: str) -> str:
    """Agent package dir. On a Databricks job `__file__` is undefined, so the job
    passes --base-dir ${workspace.file_path}."""
    if base_dir:
        return os.path.join(base_dir, "agents", "rag_agent")
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        return os.path.join(os.getcwd(), "agents", "rag_agent")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--model-name", default="mygpt_rag_agent")
    ap.add_argument("--endpoint-name", default="mygpt-rag-agent")
    ap.add_argument("--vs-endpoint", default="mygpt-vs")
    # Phase 1b: the agent's own embedding calls are reached through the Unity AI Gateway
    # endpoint, not the bge-m3 model endpoint directly. agent.py reads MYGPT_EMBED_ENDPOINT.
    ap.add_argument("--embed-endpoint", default="mygpt-embeddings")
    # The Vector Search index's managed-embedding model, invoked at query time with the
    # agent's identity — declared as a resource so retrieval has CAN QUERY on it.
    ap.add_argument("--vs-embed-endpoint", default="mygpt-bge-m3")
    # Phase 2: one shared experiment holds the LoggedModels, the eval runs (eval/), and
    # the endpoint's production traces (Agent Framework logs them to the model's experiment).
    # Declared as a DABs resource in databricks.yml.
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    here = _package_dir(args.base_dir)
    uc_model = f"{args.catalog}.{args.schema}.{args.model_name}"
    vs_index = f"{args.catalog}.{args.schema}.doc_chunks_index"
    rf_pkl = os.path.join(here, "artifacts", "random_forest_hybrid_model_base.pkl")

    mlflow.set_registry_uri("databricks-uc")
    # Serverless jobs have no default experiment — use the shared, bundle-managed one.
    mlflow.set_experiment(args.experiment)

    # Resources the serving endpoint must reach (auth provisioned by Agent Framework).
    # NOTE: two embedding endpoints are declared on purpose:
    #   * args.embed_endpoint (mygpt-embeddings, the AI Gateway) — the agent's own
    #     answer-scoring _embed() calls go through the gateway (Phase 1b);
    #   * args.vs_embed_endpoint (mygpt-bge-m3) — the Vector Search index embeds the
    #     QUERY at retrieval time through its managed-embedding model using the agent's
    #     identity, so the agent must also hold CAN QUERY on that model endpoint or
    #     retrieval fails with PermissionDenied.
    resources = [
        DatabricksVectorSearchIndex(index_name=vs_index),
        DatabricksServingEndpoint(endpoint_name=args.embed_endpoint),
        DatabricksServingEndpoint(endpoint_name=args.vs_embed_endpoint),
        DatabricksServingEndpoint(endpoint_name="databricks-claude-sonnet-5"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-4-maverick"),
        DatabricksServingEndpoint(endpoint_name="databricks-gpt-oss-120b"),
    ]

    with mlflow.start_run():
        # agent.py is self-contained (scoring/config/RF inlined) so no code_paths are
        # needed — only the random-forest pickle ships as an artifact.
        logged = mlflow.pyfunc.log_model(
            name="agent",
            python_model=os.path.join(here, "agent.py"),
            artifacts={"rf_model": rf_pkl},
            pip_requirements=os.path.join(here, "requirements.txt"),
            resources=resources,
        )

    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"Registered {uc_model} v{registered.version} (LoggedModel {logged.model_id})")

    # Deploy via Mosaic AI Agent Framework (creates/updates the serving endpoint).
    from databricks import agents

    agents.deploy(
        uc_model,
        registered.version,
        endpoint_name=args.endpoint_name,
        environment_vars={
            "MYGPT_SCHEMA": f"{args.catalog}.{args.schema}",
            "MYGPT_VS_INDEX": vs_index,
            "MYGPT_VS_ENDPOINT": args.vs_endpoint,
            "MYGPT_EMBED_ENDPOINT": args.embed_endpoint,
            "MYGPT_RF_MODEL_PATH": "/model/artifacts/rf_model/random_forest_hybrid_model_base.pkl",
            # Set MYGPT_WAREHOUSE_ID to read live config from Delta; omit for packaged defaults.
        },
        scale_to_zero_enabled=True,
    )
    print(f"Deploy requested for endpoint {args.endpoint_name}")


if __name__ == "__main__":
    main()
