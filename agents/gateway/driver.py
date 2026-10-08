"""Stand up the Unity AI Gateway endpoint in front of the bge-m3 embedding model.

Run on a Databricks cluster/notebook (or as a DABs job task). Creates/updates an
*external-model* serving endpoint (`mygpt-embeddings`) that proxies to the internal
`mygpt-bge-m3` Model Serving endpoint via the `databricks-model-serving` provider,
advertising the `llm/v1/embeddings` task. Consumers (the RAG agent, Vector Search)
address this stable gateway endpoint instead of the model endpoint directly, so the
backing model can change without touching consumers.

PREREQUISITE (one-time, manual — like the Vector Search index creation):
    The `databricks-model-serving` provider authenticates to the upstream endpoint with
    a Databricks token that has CAN QUERY on `mygpt-bge-m3`. Store it as a secret:

        databricks secrets create-scope mygpt            # if it doesn't exist
        databricks secrets put-secret mygpt gw_upstream_token --profile mygpt

    Prefer a service-principal token over a personal PAT.

Usage:
    python agents/gateway/driver.py --catalog structural_biology --schema mygpt
"""

from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    DatabricksModelServingConfig,
    EndpointCoreConfigInput,
    ExternalModel,
    ExternalModelProvider,
    ServedEntityInput,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")  # accepted for job parity
    ap.add_argument("--schema", default="mygpt")  # accepted for job parity
    ap.add_argument("--endpoint-name", default="mygpt-embeddings")
    ap.add_argument("--upstream-endpoint", default="mygpt-bge-m3")
    # Same-workspace self-reference; derived from the client host when empty.
    ap.add_argument("--workspace-url", default="")
    ap.add_argument("--token-secret", default="{{secrets/mygpt/gw_upstream_token}}")
    ap.add_argument("--base-dir", default="")  # accepted for job parity; unused here
    args = ap.parse_args()

    w = WorkspaceClient()
    workspace_url = args.workspace_url or w.config.host

    served = [
        ServedEntityInput(
            name="bge-m3-upstream",
            external_model=ExternalModel(
                name=args.upstream_endpoint,
                provider=ExternalModelProvider.DATABRICKS_MODEL_SERVING,
                task="llm/v1/embeddings",
                databricks_model_serving_config=DatabricksModelServingConfig(
                    databricks_workspace_url=workspace_url,
                    databricks_api_token=args.token_secret,
                ),
            ),
        )
    ]

    existing = [e.name for e in w.serving_endpoints.list()]
    if args.endpoint_name in existing:
        w.serving_endpoints.update_config(name=args.endpoint_name, served_entities=served)
        print(f"Updated gateway endpoint {args.endpoint_name} -> {args.upstream_endpoint}")
    else:
        w.serving_endpoints.create(
            name=args.endpoint_name,
            config=EndpointCoreConfigInput(name=args.endpoint_name, served_entities=served),
        )
        print(f"Created gateway endpoint {args.endpoint_name} -> {args.upstream_endpoint}")


if __name__ == "__main__":
    main()
