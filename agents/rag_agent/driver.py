"""Log, register, and deploy the MyGPT ResponsesAgent.

Run on a Databricks cluster/notebook (or as a DABs job task). It:
  1. logs `agent.py` as an MLflow model (code-based logging) with the RF pickle,
     the scoring/config modules, and the declared Databricks resources;
  2. registers it to Unity Catalog;
  3. deploys it to a Model Serving endpoint via Mosaic AI Agent Framework.

The embedding endpoint (`mygpt-bge-m3`), the Vector Search index, and the FMAPI
chat endpoints are declared as `resources` so Agent Framework provisions the
endpoint's auth to reach them. On-behalf-of-user auth (so AI Search retrieval runs
as the signed-in user) is validated separately — see PLAN.md risks.

Usage:
    python agents/rag_agent/driver.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse
import os
import shutil

import mlflow
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex,
)

HERE = os.path.dirname(__file__)
REF_RF_PKL = os.path.abspath(
    os.path.join(HERE, "..", "..", "reference", "MyGPT", "backend", "data",
                 "random_forest_models", "random_forest_hybrid_model_base.pkl")
)


def _stage_rf_artifact() -> str:
    """Copy the RF pickle next to the agent so it ships as a code artifact."""
    dest_dir = os.path.join(HERE, "artifacts")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, "random_forest_hybrid_model_base.pkl")
    if not os.path.exists(dest) and os.path.exists(REF_RF_PKL):
        shutil.copyfile(REF_RF_PKL, dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--model-name", default="mygpt_rag_agent")
    ap.add_argument("--endpoint-name", default="mygpt-rag-agent")
    ap.add_argument("--vs-endpoint", default="mygpt-vs")
    ap.add_argument("--embed-endpoint", default="mygpt-bge-m3")
    args = ap.parse_args()

    uc_model = f"{args.catalog}.{args.schema}.{args.model_name}"
    vs_index = f"{args.catalog}.{args.schema}.doc_chunks_index"
    _stage_rf_artifact()

    mlflow.set_registry_uri("databricks-uc")

    # Resources the serving endpoint must reach (auth provisioned by Agent Framework).
    resources = [
        DatabricksVectorSearchIndex(index_name=vs_index),
        DatabricksServingEndpoint(endpoint_name=args.embed_endpoint),
        DatabricksServingEndpoint(endpoint_name="databricks-claude-sonnet-5"),
        DatabricksServingEndpoint(endpoint_name="databricks-llama-4-maverick"),
        DatabricksServingEndpoint(endpoint_name="databricks-gpt-oss-120b"),
    ]

    with mlflow.start_run():
        logged = mlflow.pyfunc.log_model(
            name="agent",
            python_model=os.path.join(HERE, "agent.py"),
            code_paths=[
                os.path.join(HERE, "scoring.py"),
                os.path.join(HERE, "config.py"),
                os.path.join(HERE, "hi_random_forest.py"),
            ],
            artifacts={"rf_model": os.path.join(HERE, "artifacts", "random_forest_hybrid_model_base.pkl")},
            pip_requirements=os.path.join(HERE, "requirements.txt"),
            resources=resources,
        )

    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"Registered {uc_model} v{registered.version}")

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
        scale_to_zero=True,
    )
    print(f"Deploy requested for endpoint {args.endpoint_name}")


if __name__ == "__main__":
    main()
