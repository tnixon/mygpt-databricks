"""Smoke test for the deployed bge-m3 embeddings endpoint.

    python agents/embedding/smoke_test.py --profile mygpt

Verifies two call paths:
  * the raw `/invocations` path (what the RAG agent uses) returns a list of 1024-dim
    vectors — the shape `_parse_embeddings` in agent.py consumes;
  * the OpenAI embeddings client returns the OpenAI envelope with 1024-dim vectors —
    how Vector Search managed embeddings reach it. This path needs `databricks-sdk[openai]`
    and is skipped (not failed) if `openai` isn't installed locally.
"""

from __future__ import annotations

import argparse


def check_raw_vectors(w, endpoint: str, n: int = 2) -> None:
    """Raw /invocations path (the agent's path) returns a list of 1024-dim vectors."""
    inputs = ["kinase inhibitor assay", "glucocorticoid dosing policy"][:n]
    raw = w.api_client.do(
        "POST", f"/serving-endpoints/{endpoint}/invocations", body={"input": inputs}
    )
    # Both the model endpoint and the AI Gateway proxy return a bare list of vectors on
    # this path; agent.py:_parse_embeddings consumes exactly this.
    assert isinstance(raw, list) and len(raw) == n, f"[raw] expected {n} vectors, got {raw!r:.80}"
    dims = [len(v) for v in raw]
    assert all(d == 1024 for d in dims), f"[raw] expected 1024-dim, got {dims}"
    print(f"[raw] vectors={len(raw)} dims={dims}")


def check_openai_client_path(w, endpoint: str, n: int = 2) -> None:
    """OpenAI client path returns the OpenAI envelope (how Vector Search reaches it).
    Best-effort: skipped (not failed) when `openai` isn't installed locally."""
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
    ap.add_argument("--endpoint", default="mygpt-bge-m3")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    check_raw_vectors(w, args.endpoint)
    check_openai_client_path(w, args.endpoint)
    print("OK")


if __name__ == "__main__":
    main()
