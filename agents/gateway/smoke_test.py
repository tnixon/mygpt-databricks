"""Smoke test for the Unity AI Gateway embeddings endpoint (`mygpt-embeddings`).

    python agents/gateway/smoke_test.py --profile mygpt

Verifies the gateway returns correct embeddings over BOTH call paths consumers use:
  * raw /invocations (the RAG agent) returns a list of 1024-dim vectors — the shape
    agent.py:_parse_embeddings consumes;
  * the OpenAI client (how Vector Search managed embeddings reach it) returns the OpenAI
    envelope with 1024-dim vectors. The OpenAI path needs `databricks-sdk[openai]` and is
    skipped (not failed) when `openai` isn't installed locally.
"""

from __future__ import annotations

import argparse


def check_raw_vectors(w, endpoint: str, n: int = 2) -> None:
    inputs = ["kinase inhibitor assay", "glucocorticoid dosing policy"][:n]
    raw = w.api_client.do(
        "POST", f"/serving-endpoints/{endpoint}/invocations", body={"input": inputs}
    )
    assert isinstance(raw, list) and len(raw) == n, f"[raw] expected {n} vectors, got {raw!r:.80}"
    dims = [len(v) for v in raw]
    assert all(d == 1024 for d in dims), f"[raw] expected 1024-dim, got {dims}"
    print(f"[raw] vectors={len(raw)} dims={dims}")


def check_openai_client_path(w, endpoint: str, n: int = 2) -> None:
    inputs = ["kinase inhibitor assay", "glucocorticoid dosing policy"][:n]
    try:
        client = w.serving_endpoints.get_open_ai_client()
    except ImportError:
        print("[openai] skipped — install databricks-sdk[openai] to exercise this path")
        return
    resp = client.embeddings.create(model=endpoint, input=inputs)
    dims = [len(d.embedding) for d in resp.data]
    assert resp.object == "list", f"[openai] object != 'list': {resp.object!r}"
    assert len(resp.data) == n and all(d == 1024 for d in dims), f"[openai] dims={dims}"
    for i, d in enumerate(resp.data):
        assert d.index == i and d.object == "embedding", f"[openai] data[{i}] off-contract"
    print(f"[openai] object={resp.object} model={resp.model} vectors={len(resp.data)} dims={dims}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--endpoint", default="mygpt-embeddings")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    check_raw_vectors(w, args.endpoint)
    check_openai_client_path(w, args.endpoint)
    print("OK")


if __name__ == "__main__":
    main()
