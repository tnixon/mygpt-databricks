"""Tests for the Phase-5 Supervisor (routing loop, scope rules, result decoding) with a fake transport.

    uv run --no-project --with-requirements eval/requirements.txt pytest -q agents/supervisor/test_supervisor.py
"""

from __future__ import annotations

import json

import mlflow
import pytest
from mlflow.types.responses import ResponsesAgentRequest

import supervisor_agent as sa

mlflow.tracing.disable()


def tool_call(name: str, args: dict, cid: str = "c1") -> dict:
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def rag_response(answer: str = "ABL1 is expressed in Sf9 cells [1].", **co) -> dict:
    return {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}],
        "custom_outputs": {
            "model": "mygpt-chat", "corpus": co.get("corpus"), "qrs": 54, "ars": 86, "hi_eq": 0, "hi_rf": None,
            "trace_id": "tr-rag",
            "sources": [{"source_name": "10498895.pdf", "page": 3, "section": "Methods", "snippet": "Sf9…",
                         "chunk_id": "c-1", "context": "full chunk text"}],
        },
    }


FUNCTION_TOOLS = {"tools": [
    {"name": "structural_biology__mygpt_tools__search_pdb_by_gene", "description": "Find PDB structures by gene",
     "inputSchema": {"type": "object", "properties": {"gene": {"type": "string"}}, "required": ["gene"]}},
    {"name": "structural_biology__mygpt_tools__list_corpora", "description": "List corpora",
     "inputSchema": {"type": "object", "properties": {}}},
]}


class FakeTransport:
    """Scripted LLM turns; records every RAG / MCP call."""

    def __init__(self, llm_turns: list, rag: dict | None = None, mcp: dict | None = None):
        self.llm_turns = list(llm_turns)
        self.rag_resp = rag or rag_response()
        self.mcp_results = mcp or {}
        self.chats: list[dict] = []
        self.rag_calls: list[dict] = []
        self.mcp_calls: list[tuple] = []

    def chat(self, messages, tools, tool_choice="auto"):
        self.chats.append({"messages": json.loads(json.dumps(messages)), "tools": tools, "tool_choice": tool_choice})
        turn = self.llm_turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    def rag(self, body):
        self.rag_calls.append(body)
        return self.rag_resp

    def mcp(self, path, method, params=None):
        self.mcp_calls.append((path, method, params))
        key = method if method == "tools/list" else (params or {}).get("name")
        result = self.mcp_results.get(key, FUNCTION_TOOLS if method == "tools/list" else {})
        return result.pop(0) if isinstance(result, list) else result


def ask(transport, question="What expression system was used for EPHA8?", history=None, **ci):
    agent = sa.MyGPTSupervisor(transport=transport, sleep=lambda s: None)
    turns = (history or []) + [{"role": "user", "content": question}]
    return agent.predict(ResponsesAgentRequest(input=turns, custom_inputs=ci))


def answer_text(resp) -> str:
    return resp.output[0].content[0]["text"]


# --------------------------------------------------------------------------- routing
def test_documents_route_is_verbatim_and_single_hop():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "EPHA8 expression system?"})]}])
    resp = ask(t, corpus="kinase_literature", k=5, model="sonnet")
    co = resp.custom_outputs
    assert answer_text(resp) == "ABL1 is expressed in Sf9 cells [1]."  # the RAG answer, untouched
    assert len(t.chats) == 1  # no synthesis hop
    assert co["route"] == ["documents"] and co["scores_scope"] == "answer"
    assert (co["qrs"], co["ars"], co["hi_eq"], co["trace_id"]) == (54, 86, 0, "tr-rag")
    # single-turn: the RAG agent gets the user's exact question, not the LLM's rewrite
    body = t.rag_calls[0]
    assert body["input"][0]["content"] == "What expression system was used for EPHA8?"
    assert body["custom_inputs"]["include_context"] is True
    assert body["custom_inputs"]["corpus"] == "kinase_literature" and body["custom_inputs"]["k"] == 5
    # full chunk text is for the trace only, not the app payload
    assert "context" not in co["sources"][0] and co["sources"][0]["source_name"] == "10498895.pdf"


def test_multi_turn_uses_the_standalone_rewrite():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "Which assay was used for EPHA8?"})]}])
    ask(t, question="and which assay?", history=[{"role": "user", "content": "EPHA8 expression system?"},
                                                  {"role": "assistant", "content": "Sf9 cells."}])
    assert t.rag_calls[0]["input"][0]["content"] == "Which assay was used for EPHA8?"
    assert [m["role"] for m in t.chats[0]["messages"]] == ["system", "user", "assistant", "user"]


def test_forced_docs_route_skips_the_llm():
    t = FakeTransport([])
    resp = ask(t, route="docs")
    assert not t.chats and len(t.rag_calls) == 1
    assert resp.custom_outputs["scores_scope"] == "answer"


def test_router_failure_degrades_to_documents():
    t = FakeTransport([RuntimeError("429 rate limited")])
    resp = ask(t)
    assert answer_text(resp).startswith("ABL1 is expressed")
    assert resp.custom_outputs["route"] == ["documents"] and "429" in resp.custom_outputs["error"]


def test_direct_answer_without_tools():
    t = FakeTransport([{"content": [{"type": "reasoning", "summary": []}, {"type": "text", "text": "Hello!"}]}])
    resp = ask(t, question="hi")
    assert answer_text(resp) == "Hello!"
    assert resp.custom_outputs["route"] == ["direct"] and resp.custom_outputs["scores_scope"] is None


def test_multi_tool_answer_is_synthesized_and_scoped():
    t = FakeTransport([
        {"tool_calls": [tool_call("search_documents", {"query": "ABL1 expression system"}, "c1"),
                        tool_call("search_pdb_by_gene", {"gene": "ABL1"}, "c2")]},
        {"content": "Sf9 cells [1]; structures include 2HYY."},
    ], mcp={"structural_biology__mygpt_tools__search_pdb_by_gene": {"content": [{"type": "text", "text": json.dumps(
        {"is_truncated": False, "columns": ["output"],
         "rows": [[json.dumps({"total": 74, "entries": [{"rcsb_id": "2HYY"}]})]]})}]}})
    resp = ask(t)
    co = resp.custom_outputs
    assert answer_text(resp) == "Sf9 cells [1]; structures include 2HYY."
    assert co["route"] == ["documents", "tools"] and co["scores_scope"] == "documents_part"
    tool_msgs = [m for m in t.chats[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]
    assert json.loads(tool_msgs[1]["content"]) == {"total": 74, "entries": [{"rcsb_id": "2HYY"}]}
    assert t.mcp_calls[-1][2] == {"name": "structural_biology__mygpt_tools__search_pdb_by_gene", "arguments": {"gene": "ABL1"}}


def test_step_budget_ends_with_a_tool_free_answer():
    loop = {"tool_calls": [tool_call("list_corpora", {})]}
    t = FakeTransport([loop] * sa.MAX_STEPS + [{"content": "Here are the corpora."}])
    resp = ask(t)
    assert answer_text(resp) == "Here are the corpora."
    assert len(t.chats) == sa.MAX_STEPS + 1 and t.chats[-1]["tools"] is None


# --------------------------------------------------------------------------- scope rules
def test_llm_cannot_widen_the_app_scope():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {
        "query": "q", "corpus": "biomed", "documents": ["a.pdf", "other.pdf"], "section": "Results"})]}])
    ask(t, corpus="kinase_literature", focus={"documents": ["a.pdf", "b.pdf"], "section": "Methods"})
    rci = t.rag_calls[0]["custom_inputs"]
    assert rci["corpus"] == "kinase_literature"  # app corpus wins
    assert rci["focus"] == {"documents": ["a.pdf"], "section": "Methods"}  # narrowed, never widened


def test_llm_may_narrow_the_all_corpora_scope():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q", "corpus": "biomed"})]}])
    ask(t)
    assert t.rag_calls[0]["custom_inputs"]["corpus"] == "biomed"
    spec = next(s for s in t.chats[0]["tools"] if s["function"]["name"] == "search_documents")
    assert "corpus" in spec["function"]["parameters"]["properties"]


def test_upload_library_sources_are_forwarded_and_corpus_choice_hidden():
    sources = [{"chunk_text": "x", "similarity": 0.8}]
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q", "corpus": "biomed"})]}])
    ask(t, sources=sources)
    rci = t.rag_calls[0]["custom_inputs"]
    assert rci["sources"] == sources and "corpus" not in rci
    spec = next(s for s in t.chats[0]["tools"] if s["function"]["name"] == "search_documents")
    assert "corpus" not in spec["function"]["parameters"]["properties"]


# --------------------------------------------------------------------------- actions
LIBS = [{"id": "11", "name": "My FR policies", "kind": "upload", "visibility": "private"},
        {"id": "1", "name": "All governed documents", "kind": "corpus", "visibility": "public"}]


def test_actions_are_proposed_never_executed():
    t = FakeTransport([
        {"tool_calls": [tool_call("propose_library_action", {"action": "create_library", "library_name": "Kinases"}, "c1"),
                        tool_call("propose_library_action", {"action": "add_document_from_url",
                                                             "url": "https://example.org/p.pdf"}, "c2")]},
        {"content": "I've proposed creating Kinases and adding the PDF; please confirm."},
    ])
    resp = ask(t, question="make a Kinases library with https://example.org/p.pdf", libraries=LIBS, allow_actions=True)
    acts = resp.custom_outputs["actions"]
    assert [a["action"] for a in acts] == ["create_library", "add_document_from_url"]
    assert acts[0]["visibility"] == "private" and acts[1]["library_id"] is None  # = the library proposed above
    assert not t.rag_calls and all(m[1] == "tools/list" for m in t.mcp_calls)  # nothing executed
    assert resp.custom_outputs["route"] == ["libraries"]


def test_action_validation_errors_reach_the_llm():
    t = FakeTransport([
        {"tool_calls": [tool_call("propose_library_action", {"action": "reingest", "library_id": "1"}, "c1"),
                        tool_call("propose_library_action", {"action": "drop_table"}, "c2"),
                        {"id": "c3", "type": "function",
                         "function": {"name": "propose_library_action", "arguments": "{not json"}}]},
        {"content": "I can't do that."},
    ])
    resp = ask(t, libraries=LIBS, allow_actions=True)
    assert resp.custom_outputs["actions"] == []
    errors = [json.loads(m["content"])["error"] for m in t.chats[1]["messages"] if m["role"] == "tool"]
    assert "upload libraries" in errors[0] and "unknown action" in errors[1] and "malformed" in errors[2]
    assert [c["ok"] for c in resp.custom_outputs["tool_calls"]] == [False, False, False]


def test_action_and_library_tools_need_app_opt_in():
    t = FakeTransport([{"content": "ok"}])
    ask(t)
    names = {s["function"]["name"] for s in t.chats[0]["tools"]}
    assert "propose_library_action" not in names and "list_my_libraries" not in names
    assert {"search_documents", "search_pdb_by_gene", "list_corpora"} <= names


# --------------------------------------------------------------------------- Genie
def genie_msg(status, rows=None):
    content = {"textAttachments": ["2 CMGC kinases have papers."], "queryAttachments": []}
    if rows is not None:
        content["queryAttachments"] = [{"query": "SELECT kinase_id, papers FROM …", "description": "CMGC kinases",
                                        "statement_response": {
                                            "manifest": {"schema": {"columns": [{"name": "kinase_id"}, {"name": "papers"}]},
                                                         "total_row_count": len(rows)},
                                            "result": {"data_array": rows}}}]
    return {"structuredContent": {"status": status, "conversationId": "conv", "messageId": "msg", "content": content}}


def test_genie_polls_until_complete_and_decodes_rows(monkeypatch):
    monkeypatch.setattr(sa, "GENIE_SPACE_ID", "space1")
    rows = [{"values": [{"string_value": "MAPK14"}, {"string_value": "6"}]}, {"values": [{"string_value": "CDK2"}, {"null_value": None}]}]
    t = FakeTransport(
        [{"tool_calls": [tool_call("query_structured_data", {"question": "CMGC kinases with papers?"})]},
         {"content": "MAPK14 has 6 papers."}],
        mcp={"query_space_space1": genie_msg("ASKING_AI"),
             "poll_response_space1": [genie_msg("EXECUTING_QUERY"), genie_msg("COMPLETED", rows)]},
    )
    resp = ask(t, question="Which CMGC kinases have papers?", route="genie")
    g = resp.custom_outputs["genie"]
    assert g["status"] == "COMPLETED" and g["columns"] == ["kinase_id", "papers"]
    assert g["rows"] == [["MAPK14", "6"], ["CDK2", None]] and g["sql"].startswith("SELECT")
    assert t.chats[0]["tool_choice"] == "required"
    assert [s["function"]["name"] for s in t.chats[0]["tools"]] == ["query_structured_data"]
    polls = [c for c in t.mcp_calls if c[2] and c[2]["name"] == "poll_response_space1"]
    assert len(polls) == 2 and polls[0][2]["arguments"] == {"conversation_id": "conv", "message_id": "msg"}
    assert resp.custom_outputs["route"] == ["genie"] and resp.custom_outputs["scores_scope"] is None


# --------------------------------------------------------------------------- decoding
def test_decode_function_results():
    assert sa.decode_function_result({"columns": ["output"], "rows": [['{"a": 1}']]}) == {"a": 1}
    assert sa.decode_function_result({"columns": ["output"], "rows": [["plain"]]}) == "plain"
    table = {"columns": ["corpus", "documents"], "rows": [["biomed", 397], ["kinase_literature", 91]], "is_truncated": False}
    assert sa.decode_function_result(table) == [{"corpus": "biomed", "documents": 397},
                                                {"corpus": "kinase_literature", "documents": 91}]
    assert sa.decode_function_result({**table, "is_truncated": True})["truncated"] is True


def test_function_tools_listing_failure_is_not_fatal():
    class Down(FakeTransport):
        def mcp(self, path, method, params=None):
            raise RuntimeError("MCP unavailable")

    t = Down([{"tool_calls": [tool_call("search_documents", {"query": "q"})]}])
    resp = ask(t)
    assert resp.custom_outputs["route"] == ["documents"] and "MCP unavailable" in resp.custom_outputs["error"]


@pytest.mark.parametrize("content,expected", [
    ("plain", "plain"), (None, ""),
    ([{"type": "reasoning", "summary": [{"text": "think"}]}, {"type": "text", "text": "answer"}], "answer"),
])
def test_message_text(content, expected):
    assert sa.message_text(content) == expected
