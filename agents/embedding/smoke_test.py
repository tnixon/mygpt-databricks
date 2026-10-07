"""Smoke test for the deployed bge-m3 embeddings endpoint.

    python agents/embedding/smoke_test.py --profile mygpt

Verifies the endpoint returns 1024-dim vectors via the OpenAI embeddings client,
which is exactly how Vector Search managed embeddings and the agent will call it.
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--endpoint", default="mygpt-bge-m3")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    client = w.serving_endpoints.get_open_ai_client()
    resp = client.embeddings.create(
        model=args.endpoint, input=["kinase inhibitor assay", "glucocorticoid dosing policy"]
    )
    dims = [len(d.embedding) for d in resp.data]
    print(f"vectors: {len(resp.data)}  dims: {dims}")
    assert len(resp.data) == 2, "expected 2 vectors"
    assert all(d == 1024 for d in dims), f"expected 1024-dim, got {dims}"
    print("OK")


if __name__ == "__main__":
    main()
