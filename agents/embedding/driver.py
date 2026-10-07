"""Log + register + deploy BAAI/bge-m3 to the `mygpt-bge-m3` embeddings endpoint.

Run on a Databricks cluster/notebook (or as a DABs job task). The endpoint must
advertise the embeddings task (`llm/v1/embeddings`) so Vector Search can use it as
a managed embedding model. GPU small; scale-to-zero for the POC.

Usage:
    python agents/embedding/driver.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse
import os

import mlflow
from mlflow.models import infer_signature

HERE = os.path.dirname(__file__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--model-name", default="bge_m3")
    ap.add_argument("--endpoint-name", default="mygpt-bge-m3")
    args = ap.parse_args()

    uc_model = f"{args.catalog}.{args.schema}.{args.model_name}"
    mlflow.set_registry_uri("databricks-uc")

    example = {"input": ["kinase inhibitor assay"]}
    signature = infer_signature(
        example,
        {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 1024}]},
    )

    with mlflow.start_run():
        logged = mlflow.pyfunc.log_model(
            name="bge_m3",
            python_model=os.path.join(HERE, "bge_m3_model.py"),
            signature=signature,
            input_example=example,
            pip_requirements=os.path.join(HERE, "requirements.txt"),
        )

    registered = mlflow.register_model(model_uri=logged.model_uri, name=uc_model)
    print(f"Registered {uc_model} v{registered.version}")

    # Create/update the serving endpoint with the embeddings task on GPU small.
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    w = WorkspaceClient()
    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=uc_model,
                entity_version=registered.version,
                workload_type="GPU_SMALL",
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ]
    )
    existing = [e.name for e in w.serving_endpoints.list()]
    if args.endpoint_name in existing:
        w.serving_endpoints.update_config(name=args.endpoint_name, served_entities=config.served_entities)
        print(f"Updated endpoint {args.endpoint_name}")
    else:
        w.serving_endpoints.create(name=args.endpoint_name, config=config, task="llm/v1/embeddings")
        print(f"Created endpoint {args.endpoint_name}")


if __name__ == "__main__":
    main()
