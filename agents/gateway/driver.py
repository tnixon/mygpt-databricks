"""Stand up a Unity AI Gateway endpoint in front of a MyGPT model.

Run on a Databricks cluster/notebook (or as a DABs job task). Creates/updates an
*external-model* serving endpoint that proxies to an upstream Model Serving endpoint via
the `databricks-model-serving` provider. Consumers address the stable gateway endpoint
instead of the model endpoint directly, so the backing model can change without touching
consumers.

  --task embeddings (Phase 1b)  mygpt-embeddings -> mygpt-bge-m3        (llm/v1/embeddings)
  --task chat       (Phase 5)   mygpt-chat       -> databricks-claude-sonnet-5 (llm/v1/chat)
  --task translate  (Phase 6)   mygpt-translate  -> databricks-gemma-3-12b     (llm/v1/chat)

translate fronts the RAG agent's query/answer translation (gemma-3-12b is the base model
of MyGPT's translategemma), so the translator can be swapped without redeploying the agent.

For chat and translate the driver also sets the AI Gateway features (usage tracking, an inference table
in <catalog>.<schema>, a per-user rate limit). Passing --azure-endpoint adds St. Jude's
Azure AI Foundry deployment as a second served entity at 0% traffic with fallback on, so
FMAPI stays primary and Azure takes over on upstream errors. Phase-5 spike (2026-10-15):
tools / tool_calls pass through this provider unchanged.

PREREQUISITE (one-time, manual — like the Vector Search index creation):
    The `databricks-model-serving` provider authenticates to the upstream endpoint with
    a Databricks token that has CAN QUERY on it. Store it as a secret:

        databricks secrets create-scope mygpt            # if it doesn't exist
        databricks secrets put-secret mygpt gw_upstream_token --profile mygpt

    Prefer a service-principal token over a personal PAT. The Azure route needs the
    Foundry key as well: databricks secrets put-secret mygpt azure_foundry_key.

Usage:
    python agents/gateway/driver.py --task embeddings
    python agents/gateway/driver.py --task translate
    python agents/gateway/driver.py --task chat [--azure-endpoint https://<res>.openai.azure.com --azure-deployment <name>]
"""

from __future__ import annotations

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayInferenceTableConfig,
    AiGatewayRateLimit,
    AiGatewayRateLimitKey,
    AiGatewayRateLimitRenewalPeriod,
    AiGatewayUsageTrackingConfig,
    DatabricksModelServingConfig,
    EndpointCoreConfigInput,
    ExternalModel,
    ExternalModelProvider,
    FallbackConfig,
    OpenAiConfig,
    Route,
    ServedEntityInput,
    TrafficConfig,
)

TASKS = {
    # task: (default endpoint, default upstream, served-entity name, external-model task)
    "embeddings": ("mygpt-embeddings", "mygpt-bge-m3", "bge-m3-upstream", "llm/v1/embeddings"),
    "chat": ("mygpt-chat", "databricks-claude-sonnet-5", "sonnet", "llm/v1/chat"),
    "translate": ("mygpt-translate", "databricks-gemma-3-12b", "gemma-3-12b", "llm/v1/chat"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), default="embeddings")
    ap.add_argument("--catalog", default="structural_biology")  # chat: inference-table location
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--endpoint-name", default="")
    ap.add_argument("--upstream-endpoint", default="")
    # Same-workspace self-reference; derived from the client host when empty.
    ap.add_argument("--workspace-url", default="")
    # Secret NAMES only; the endpoint resolves {{secrets/<scope>/<key>}} itself. Passed as plain
    # names because a job parameter containing {{...}} is read as a dynamic value reference.
    ap.add_argument("--secret-scope", default="mygpt")
    ap.add_argument("--token-secret-key", default="gw_upstream_token")
    # chat / translate: per-user calls per minute across all consumers of the endpoint.
    ap.add_argument("--user-calls-per-minute", type=int, default=60)
    # chat only: the Azure AI Foundry route (Azure OpenAI-compatible deployment). Off when empty.
    ap.add_argument("--azure-endpoint", default="")
    ap.add_argument("--azure-deployment", default="")
    ap.add_argument("--azure-api-version", default="2024-10-21")
    ap.add_argument("--azure-key-secret-key", default="azure_foundry_key")
    ap.add_argument("--base-dir", default="")  # accepted for job parity; unused here
    args = ap.parse_args()

    default_endpoint, default_upstream, entity_name, task = TASKS[args.task]
    endpoint_name = args.endpoint_name or default_endpoint
    upstream = args.upstream_endpoint or default_upstream

    w = WorkspaceClient()
    workspace_url = args.workspace_url or w.config.host

    served = [
        ServedEntityInput(
            name=entity_name,
            external_model=ExternalModel(
                name=upstream,
                provider=ExternalModelProvider.DATABRICKS_MODEL_SERVING,
                task=task,
                databricks_model_serving_config=DatabricksModelServingConfig(
                    databricks_workspace_url=workspace_url,
                    databricks_api_token=f"{{{{secrets/{args.secret_scope}/{args.token_secret_key}}}}}",
                ),
            ),
        )
    ]
    traffic = None
    azure = args.task == "chat" and bool(args.azure_endpoint)
    if azure:
        if not args.azure_deployment:
            raise SystemExit("--azure-deployment is required with --azure-endpoint")
        served.append(ServedEntityInput(
            name="azure-foundry",
            external_model=ExternalModel(
                name=args.azure_deployment,
                provider=ExternalModelProvider.OPENAI,
                task=task,
                openai_config=OpenAiConfig(
                    openai_api_type="azure",
                    openai_api_base=args.azure_endpoint,
                    openai_deployment_name=args.azure_deployment,
                    openai_api_version=args.azure_api_version,
                    openai_api_key=f"{{{{secrets/{args.secret_scope}/{args.azure_key_secret_key}}}}}",
                ),
            ),
        ))
        # FMAPI serves everything; the Azure entity only receives fallback traffic.
        traffic = TrafficConfig(routes=[
            Route(served_model_name=entity_name, traffic_percentage=100),
            Route(served_model_name="azure-foundry", traffic_percentage=0),
        ])

    existing = [e.name for e in w.serving_endpoints.list()]
    if endpoint_name in existing:
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name, served_entities=served, traffic_config=traffic
        )
        print(f"Updated gateway endpoint {endpoint_name} -> {upstream}" + (" (+ azure-foundry)" if azure else ""))
    else:
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(name=endpoint_name, served_entities=served, traffic_config=traffic),
        )
        print(f"Created gateway endpoint {endpoint_name} -> {upstream}" + (" (+ azure-foundry)" if azure else ""))

    if task == "llm/v1/chat":
        w.serving_endpoints.put_ai_gateway(
            name=endpoint_name,
            usage_tracking_config=AiGatewayUsageTrackingConfig(enabled=True),
            inference_table_config=AiGatewayInferenceTableConfig(
                enabled=True, catalog_name=args.catalog, schema_name=args.schema,
                table_name_prefix=endpoint_name.replace("-", "_"),
            ),
            rate_limits=[AiGatewayRateLimit(
                calls=args.user_calls_per_minute, key=AiGatewayRateLimitKey.USER,
                renewal_period=AiGatewayRateLimitRenewalPeriod.MINUTE,
            )],
            fallback_config=FallbackConfig(enabled=azure),
        )
        print(f"AI Gateway on {endpoint_name}: usage tracking, inference table "
              f"{args.catalog}.{args.schema}.{endpoint_name.replace('-', '_')}_payload, "
              f"{args.user_calls_per_minute} calls/min/user, fallback={'on' if azure else 'off'}")


if __name__ == "__main__":
    main()
