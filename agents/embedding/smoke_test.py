"""Smoke test for the deployed bge-m3 embeddings endpoint.

    python agents/embedding/smoke_test.py --profile mygpt

Verifies the endpoint returns 1024-dim vectors via the OpenAI embeddings client, and
pins the raw invocations response to the OpenAI embeddings envelope — the contract the
Phase 1b AI Gateway proxy (`mygpt-embeddings`) forwards unchanged, and the shape Vector
Search managed embeddings and the agent depend on.
"""

from __future__ import annotations

import argparse


def _check_openai_envelope(resp: dict) -> None:
    assert resp.get("object") == "list", f"expected object='list', got {resp.get('object')!r}"
    assert "model" in resp, "missing top-level 'model'"
    data = resp.get("data")
    assert isinstance(data, list) and len(data) == 2, f"expected 2 data items, got {data!r}"
    for i, d in enumerate(data):
        assert d.get("object") == "embedding", f"data[{i}].object != 'embedding'"
        assert d.get("index") == i, f"data[{i}].index != {i}"
        emb = d.get("embedding")
        assert isinstance(emb, list) and len(emb) == 1024, (
            f"data[{i}].embedding expected 1024-dim list, got len "
            f"{len(emb) if isinstance(emb, list) else type(emb)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--endpoint", default="mygpt-bge-m3")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    inputs = ["kinase inhibitor assay", "glucocorticoid dosing policy"]

    # Raw invocations path — assert the full OpenAI envelope the gateway must preserve.
    raw = w.api_client.do(
        "POST", f"/serving-endpoints/{args.endpoint}/invocations", body={"input": inputs}
    )
    _check_openai_envelope(raw)
    print(f"[raw] object={raw['object']} model={raw['model']} vectors={len(raw['data'])}")

    # OpenAI client path — how Vector Search managed embeddings and the agent call it.
    client = w.serving_endpoints.get_open_ai_client()
    resp = client.embeddings.create(model=args.endpoint, input=inputs)
    dims = [len(d.embedding) for d in resp.data]
    print(f"[openai] vectors: {len(resp.data)}  dims: {dims}")
    assert len(resp.data) == 2, "expected 2 vectors"
    assert all(d == 1024 for d in dims), f"expected 1024-dim, got {dims}"
    print("OK")


if __name__ == "__main__":
    main()
