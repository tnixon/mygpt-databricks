"""Smoke test for the Unity AI Gateway endpoints.

    python agents/gateway/smoke_test.py --profile mygpt                 # mygpt-embeddings
    python agents/gateway/smoke_test.py --profile mygpt --task chat     # mygpt-chat (Phase 5)

Chat: the Supervisor relies on native tool calling through the gateway, so the check is a
two-turn contract — the model must answer a tool-shaped question with a `tool_calls`
message, then turn the `role: tool` result into a plain answer.

Embeddings: verifies the gateway returns correct embeddings over BOTH call paths consumers use:
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


SEARCH_TOOL = {"type": "function", "function": {
    "name": "search_pdb", "description": "Search the RCSB Protein Data Bank for 3D structures",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
}}


def check_chat_tool_calling(w, endpoint: str) -> None:
    messages = [{"role": "user", "content": "Which PDB structures exist for ABL1? Use the tool."}]
    first = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations",
                            body={"messages": messages, "tools": [SEARCH_TOOL], "max_tokens": 200})
    choice = first["choices"][0]
    calls = (choice.get("message") or {}).get("tool_calls") or []
    assert choice.get("finish_reason") == "tool_calls" and calls, f"[chat] no tool call: {choice!r:.200}"
    assert calls[0]["function"]["name"] == "search_pdb", f"[chat] wrong tool: {calls[0]!r:.200}"
    print(f"[chat] tool_call {calls[0]['function']['name']}({calls[0]['function']['arguments']})")
    messages += [
        {"role": "assistant", "content": None, "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"entries": [{"rcsb_id": "2HYY"}]}'},
    ]
    second = w.api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations",
                             body={"messages": messages, "tools": [SEARCH_TOOL], "max_tokens": 200})
    content = second["choices"][0]["message"].get("content")
    assert content, f"[chat] empty answer after tool result: {second!r:.200}"
    print(f"[chat] answer after tool result: {str(content)[:80]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--task", choices=["embeddings", "chat"], default="embeddings")
    ap.add_argument("--endpoint", default="")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    if args.task == "chat":
        check_chat_tool_calling(w, args.endpoint or "mygpt-chat")
    else:
        check_raw_vectors(w, args.endpoint or "mygpt-embeddings")
        check_openai_client_path(w, args.endpoint or "mygpt-embeddings")
    print("OK")


if __name__ == "__main__":
    main()
