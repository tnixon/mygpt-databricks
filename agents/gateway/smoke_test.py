"""Smoke test for the Unity AI Gateway embeddings endpoint (`mygpt-embeddings`).

    python agents/gateway/smoke_test.py --profile mygpt

Verifies the gateway returns the OpenAI embeddings envelope and 1024-dim vectors over
BOTH call paths consumers use: the raw REST invocations path (the RAG agent) and the
OpenAI client (how Vector Search managed embeddings reach it). The raw-path assertions
pin the exact contract the external-model proxy must preserve from the upstream model.
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
    ap.add_argument("--endpoint", default="mygpt-embeddings")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    inputs = ["kinase inhibitor assay", "glucocorticoid dosing policy"]

    # 1) Raw REST invocations path (the RAG agent's path) — pin the full OpenAI contract.
    raw = w.api_client.do(
        "POST", f"/serving-endpoints/{args.endpoint}/invocations", body={"input": inputs}
    )
    _check_openai_envelope(raw)
    print(f"[raw] object={raw['object']} model={raw['model']} vectors={len(raw['data'])}")

    # 2) OpenAI client path (how Vector Search managed embeddings reach it).
    client = w.serving_endpoints.get_open_ai_client()
    resp = client.embeddings.create(model=args.endpoint, input=inputs)
    dims = [len(d.embedding) for d in resp.data]
    assert len(resp.data) == 2 and all(d == 1024 for d in dims), f"openai path dims={dims}"
    print(f"[openai] vectors={len(resp.data)} dims={dims}")

    print("OK")


if __name__ == "__main__":
    main()
