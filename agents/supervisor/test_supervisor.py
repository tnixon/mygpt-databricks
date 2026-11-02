"""Tests for the Phase-5 Supervisor (routing loop, scope rules, result decoding) with a fake transport.

    uv run --no-project --with-requirements eval/requirements.txt pytest -q agents/supervisor/test_supervisor.py
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import langsmith
import mlflow
import pytest
from mlflow.types.responses import ResponsesAgentRequest

import supervisor_agent as sa

mlflow.tracing.disable()
os.environ["LANGSMITH_TRACING"] = "false"  # a developer's .env must never send test traces


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

    def __init__(self, llm_turns: list, rag=None, mcp: dict | None = None):
        self.llm_turns = list(llm_turns)
        self.rag_resp = rag or rag_response()  # a response, or body -> response
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
        return self.rag_resp(body) if callable(self.rag_resp) else self.rag_resp

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


def test_language_inputs_forwarded_and_translations_passed_through():
    rag = rag_response("Oui [1].", corpus="health_policies_fr")
    rag["custom_outputs"].update({"documents_language": "french", "translated_query": "Le plan ... ?",
                                  "translated_answer": "Oui [1].", "translate_error": None,
                                  "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"})
    t = FakeTransport([], rag=rag)
    resp = ask(t, route="docs", corpus="health_policies_fr", translate=False, query_language="english")
    ci = t.rag_calls[0]["custom_inputs"]
    assert ci["translate"] is False and ci["query_language"] == "english"
    co = resp.custom_outputs
    assert co["documents_language"] == "french" and co["translated_query"] == "Le plan ... ?"
    assert co["reranker"].startswith("cross-encoder/mmarco")


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


def test_only_read_tools_are_offered():
    t = FakeTransport([{"content": "ok"}])
    ask(t, libraries=LIBS)
    names = {s["function"]["name"] for s in t.chats[0]["tools"]}
    assert "propose_library_action" not in names and "list_my_libraries" not in names
    assert {"search_documents", "search_pdb_by_gene", "list_corpora"} <= names
    assert "actions" not in ask(FakeTransport([{"content": "ok"}])).custom_outputs


# --------------------------------------------------------------------------- libraries
LIBS = [
    {"id": "lib-kin", "name": "BlueSky kinase literature", "kind": "corpus", "corpus": "kinase_literature",
     "dataset": "kinase_literature", "documents_language": "english", "description": "BlueSky kinase papers."},
    {"id": "lib-bio", "name": "Biomedical seed (BioASQ)", "kind": "corpus", "corpus": "biomed",
     "dataset": "biomed", "documents_language": "english"},
    {"id": "lib-up", "name": "My notes", "kind": "upload", "dataset": "mygpt", "documents_language": "english",
     "sources": [{"chunk_id": "u1", "chunk_text": "x", "similarity": 0.8}]},
]


def rag_by_library(body):
    """Per-library RAG responses: two sources each, keyed by corpus (upload = provided)."""
    ci = body["custom_inputs"]
    name = ci.get("corpus") or "upload"
    scores = {"kinase_literature": 61, "biomed": 40, "upload": 70}[name]
    resp = rag_response(f"{name} says A [1] and B [2], see [1, 2].", corpus=ci.get("corpus"))
    resp["custom_outputs"].update({"qrs": scores, "trace_id": f"tr-{name}", "sources": [
        {"source_name": f"{name}-a.pdf", "chunk_id": "a", "context": "ctx"},
        {"source_name": f"{name}-b.pdf", "chunk_id": "b", "context": "ctx"}]})
    return resp


def test_system_prompt_lists_the_selected_libraries_with_keys():
    t = FakeTransport([{"content": "ok"}])
    ask(t, libraries=LIBS)
    system = t.chats[0]["messages"][0]["content"]
    assert "L1: 'BlueSky kinase literature' (governed collection, english) - BlueSky kinase papers." in system
    assert "L3: 'My notes' (the user's uploaded documents, english)" in system
    spec = next(s for s in t.chats[0]["tools"] if s["function"]["name"] == "search_documents")
    props = spec["function"]["parameters"]["properties"]
    assert props["libraries"]["items"]["enum"] == ["L1", "L2", "L3"] and "corpus" not in props


def test_one_chosen_library_is_verbatim_with_its_scores():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q", "libraries": ["L1"]})]}], rag=rag_by_library)
    resp = ask(t, libraries=LIBS, k=4)
    co = resp.custom_outputs
    assert len(t.chats) == 1 and len(t.rag_calls) == 1
    rci = t.rag_calls[0]["custom_inputs"]
    assert rci["corpus"] == "kinase_literature" and rci["dataset"] == "kinase_literature" and rci["k"] == 4
    assert answer_text(resp) == "kinase_literature says A [1] and B [2], see [1, 2]."
    assert co["scores_scope"] == "answer" and co["qrs"] == 61 and co["trace_id"] == "tr-kinase_literature"
    assert [s["library_id"] for s in co["sources"]] == ["lib-kin", "lib-kin"]
    assert co["libraries_searched"][0]["library_id"] == "lib-kin" and co["libraries_searched"][0]["first_source"] == 1


def test_several_libraries_are_merged_with_global_citations():
    t = FakeTransport([
        {"tool_calls": [tool_call("search_documents", {"query": "q", "libraries": ["L2", "L3"]}, "c1")]},
        {"content": "Merged [1] and [3]."},
    ], rag=rag_by_library)
    resp = ask(t, libraries=LIBS)
    co = resp.custom_outputs
    assert answer_text(resp) == "Merged [1] and [3]." and len(t.chats) == 2
    # upload library: its pre-retrieved hits are forwarded, no corpus
    up = next(b["custom_inputs"] for b in t.rag_calls if "sources" in b["custom_inputs"])
    assert up["sources"] == LIBS[2]["sources"] and "corpus" not in up
    tool = json.loads(next(m["content"] for m in t.chats[1]["messages"] if m["role"] == "tool"))
    first, second = tool["results"]
    assert first["library"] == "Biomedical seed (BioASQ)" and [s["n"] for s in first["sources"]] == [1, 2]
    assert second["answer"] == "upload says A [3] and B [4], see [3, 4]." and [s["n"] for s in second["sources"]] == [3, 4]
    assert co["scores_scope"] == "per_library" and co["qrs"] is None
    assert [(x["library_id"], x["qrs"], x["first_source"]) for x in co["libraries_searched"]] == [
        ("lib-bio", 40, 1), ("lib-up", 70, 3)]
    assert [s["source_name"] for s in co["sources"]] == ["biomed-a.pdf", "biomed-b.pdf", "upload-a.pdf", "upload-b.pdf"]
    assert co["trace_id"] is None or co["trace_id"] == co["supervisor_trace_id"]  # no single RAG trace


def test_llm_cannot_pick_libraries_outside_the_set():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q", "libraries": ["L9", "L2"]})]}], rag=rag_by_library)
    ask(t, libraries=LIBS)
    assert [b["custom_inputs"].get("corpus") for b in t.rag_calls] == ["biomed"]


def test_omitted_libraries_search_the_whole_set():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q"})]}, {"content": "All."}], rag=rag_by_library)
    resp = ask(t, libraries=LIBS)
    assert len(t.rag_calls) == 3 and len(resp.custom_outputs["libraries_searched"]) == 3


def test_numbering_continues_across_search_calls():
    t = FakeTransport([
        {"tool_calls": [tool_call("search_documents", {"query": "q1", "libraries": ["L1"]}, "c1"),
                        tool_call("list_corpora", {}, "c2")]},
        {"tool_calls": [tool_call("search_documents", {"query": "q2", "libraries": ["L2"]}, "c3")]},
        {"content": "done"},
    ], rag=rag_by_library)
    resp = ask(t, libraries=LIBS)
    co = resp.custom_outputs
    assert [x["first_source"] for x in co["libraries_searched"]] == [1, 3]
    last_tool = json.loads([m for m in t.chats[2]["messages"] if m["role"] == "tool"][-1]["content"])
    assert last_tool["answer"].startswith("biomed says A [3] and B [4]")


def test_forced_docs_route_merges_several_libraries():
    t = FakeTransport([{"content": "Merged."}], rag=rag_by_library)
    resp = ask(t, route="docs", libraries=LIBS[:2])
    assert len(t.rag_calls) == 2 and t.chats[0]["tools"] is None
    assert answer_text(resp) == "Merged." and resp.custom_outputs["scores_scope"] == "per_library"


def test_router_down_with_several_libraries_stacks_their_answers():
    t = FakeTransport([RuntimeError("503")], rag=rag_by_library)
    resp = ask(t, libraries=LIBS[:2])
    text = answer_text(resp)
    assert text.startswith("**BlueSky kinase literature**") and "biomed says A [3]" in text


def test_one_failing_library_keeps_the_others():
    def flaky(body):
        if body["custom_inputs"].get("corpus") == "biomed":
            raise RuntimeError("endpoint scaling up")
        return rag_by_library(body)

    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q", "libraries": ["L1", "L2"]})]},
                       {"content": "Kinase part only [1]."}], rag=flaky)
    co = ask(t, libraries=LIBS).custom_outputs
    errors = {x["library_id"]: x["retrieve_error"] for x in co["libraries_searched"]}
    assert errors["lib-kin"] is None and "scaling up" in errors["lib-bio"]
    assert co["tool_calls"][0]["ok"] is True and len(co["sources"]) == 2


def test_an_empty_library_set_never_widens_to_all_documents():
    t = FakeTransport([{"content": "Nothing in your libraries matches."}])
    resp = ask(t, libraries=[])
    assert not t.rag_calls and "search_documents" not in {s["function"]["name"] for s in t.chats[0]["tools"]}
    assert "unavailable" in t.chats[0]["messages"][0]["content"]
    t = FakeTransport([])
    resp = ask(t, route="docs", libraries=[])
    assert not t.rag_calls and resp.custom_outputs["tool_calls"][0]["ok"] is False


@pytest.mark.parametrize("text,offset,n,expected", [
    ("A [1], B [2] and [1, 2].", 4, 2, "A [5], B [6] and [5, 6]."),
    ("A [1] and year [2024].", 2, 3, "A [3] and year [2024]."),
    ("A [1].", 0, 1, "A [1]."),
])
def test_renumber_citations(text, offset, n, expected):
    assert sa.renumber_citations(text, offset, n) == expected


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


# --------------------------------------------------------------------------- LangSmith
def run_names(client: MagicMock) -> list[str]:
    return [c.kwargs.get("name") for c in client.create_run.call_args_list]


def test_langsmith_off_sends_nothing_and_forwards_no_parent():
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q"})]}])
    resp = ask(t)
    assert "langsmith_parent" not in t.rag_calls[0]["custom_inputs"]
    assert resp.custom_outputs["langsmith_run_id"] is None


def test_langsmith_missing_package_is_harmless(monkeypatch):
    monkeypatch.setattr(sa, "ls", None)
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q"})]}])
    resp = ask(t, route="docs")
    assert answer_text(resp) == "ABL1 is expressed in Sf9 cells [1]." and resp.custom_outputs["langsmith_run_id"] is None


def test_langsmith_on_mirrors_spans_and_links_the_rag_run():
    client = MagicMock()
    t = FakeTransport([{"tool_calls": [tool_call("search_documents", {"query": "q"})]}])
    with langsmith.tracing_context(enabled=True, client=client, project_name="test"):
        resp = ask(t, corpus="kinase_literature", langsmith_parent="not headers")
    names = run_names(client)
    assert names[0] == "mygpt-supervisor" and {"chat", "search_documents", "retrieve"} <= set(names)
    root = client.create_run.call_args_list[0].kwargs
    assert root["inputs"]["custom_inputs"] == {"corpus": "kinase_literature"}  # no langsmith_parent / sources
    assert resp.custom_outputs["langsmith_run_id"] == str(root["id"])
    # The RAG endpoint gets the retrieve run's headers, so its run nests in the same tree.
    parent = t.rag_calls[0]["custom_inputs"]["langsmith_parent"]
    assert parent["langsmith-trace"]
    child = langsmith.RunTree.from_headers(parent)
    retrieve = next(c.kwargs for c in client.create_run.call_args_list if c.kwargs["name"] == "retrieve")
    assert child.id == retrieve["id"] and child.trace_id == root["id"]


def test_langsmith_root_nests_under_a_callers_headers():
    client = MagicMock()
    with langsmith.tracing_context(enabled=True, client=client, project_name="test"):
        with langsmith.trace("caller", "chain") as caller:
            headers = caller.to_headers()
        t = FakeTransport([{"content": "Hello!"}])
        ask(t, question="hi", langsmith_parent=headers)
    root = next(c.kwargs for c in client.create_run.call_args_list if c.kwargs["name"] == "mygpt-supervisor")
    assert root["parent_run_id"] == caller.id and root["trace_id"] == caller.trace_id
