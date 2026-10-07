"""Log + register + deploy BAAI/bge-m3 to the `mygpt-bge-m3` embeddings endpoint.

Run on a Databricks cluster/notebook (or as a DABs job task). Logs bge-m3 with the
MLflow ``sentence_transformers`` flavor and ``task="llm/v1/embeddings"`` so the
serving endpoint advertises the embeddings task — the form Vector Search accepts as
a managed embedding model. GPU small; scale-to-zero for the POC.

Usage:
    python agents/embedding/driver.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse

import mlflow
import mlflow.sentence_transformers
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from sentence_transformers import SentenceTransformer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--model-name", default="bge_m3")
    ap.add_argument("--endpoint-name", default="mygpt-bge-m3")
    ap.add_argument("--base-dir", default="")  # accepted for job parity; unused here
    args = ap.parse_args()

    uc_model = f"{args.catalog}.{args.schema}.{args.model_name}"
    w = WorkspaceClient()

    mlflow.set_registry_uri("databricks-uc")
    # Serverless jobs have no default experiment — set one under the runner's home.
    me = w.current_user.me().user_name
    mlflow.set_experiment(f"/Users/{me}/mygpt_phase1")

    model = SentenceTransformer("BAAI/bge-m3")

    with mlflow.start_run():
        logged = mlflow.sentence_transformers.log_model(
            model,
            name="bge_m3",
            task="llm/v1/embeddings",
            input_example=["kinase inhibitor assay"],
        )

    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"Registered {uc_model} v{registered.version}")

    # Create/update the serving endpoint (embeddings task inferred from the model).
    served = [
        ServedEntityInput(
            entity_name=uc_model,
            entity_version=registered.version,
            workload_type="GPU_SMALL",
            workload_size="Small",
            # Must stay warm: Vector Search embeds queries through this endpoint at
            # query time, which fails if it is scaled to zero. (Cost: always-on GPU.)
            scale_to_zero_enabled=False,
        )
    ]
    existing = [e.name for e in w.serving_endpoints.list()]
    if args.endpoint_name in existing:
        w.serving_endpoints.update_config(name=args.endpoint_name, served_entities=served)
        print(f"Updated endpoint {args.endpoint_name}")
    else:
        w.serving_endpoints.create(
            name=args.endpoint_name,
            config=EndpointCoreConfigInput(name=args.endpoint_name, served_entities=served),
        )
        print(f"Created endpoint {args.endpoint_name}")


if __name__ == "__main__":
    main()
