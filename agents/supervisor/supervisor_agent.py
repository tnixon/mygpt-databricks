"""MyGPT Supervisor (Phase 5): routes each question to documents, structured data or tools.

A self-contained MLflow ``ResponsesAgent`` on its own Model Serving endpoint
(``mygpt-supervisor``). It runs a tool-calling loop on the ``mygpt-chat`` Unity AI Gateway
endpoint over these tools:

  search_documents       -> the mygpt-rag-agent endpoint (AI Search or the app's upload-library
                            hits, rerank, QRS / ARS / HI, cited answer). Unchanged.
  query_structured_data  -> the Genie space over the kinome / kinase-paper / document tables,
                            through managed MCP (asynchronous: query, then poll).
  UC-function tools      -> managed MCP over structural_biology.mygpt_tools (PubMed, RCSB PDB,
                            governed corpora). Discovered with tools/list, so a new function
                            becomes a tool without redeploying.
  list_my_libraries      -> the user's visible libraries, handed over by the app
                            (custom_inputs.libraries): the agents never touch Lakebase.
  propose_library_action -> create_library / add_document_from_url / reingest. NEVER executed
                            here: returned in custom_outputs.actions; the app runs them under the
                            user's RLS context after the user confirms.

Scores stay meaningful: when the first LLM turn routes to search_documents alone, the RAG
agent's answer and its QRS / ARS / HI are returned VERBATIM (scores_scope = "answer") with
no second LLM hop. Answers combining several tools are written by the LLM, and the RAG
scores then describe only their documents part (scores_scope = "documents_part").

custom_inputs (all optional): route (auto | docs | genie | tools), library ({id, name, kind}),
libraries, allow_actions, plus everything the RAG agent accepts (corpus, dataset, model, k,
focus, instructions, sources), which is forwarded. The LLM may NARROW the app's corpus /
documents / section scope but never widen it.

Like agents/rag_agent/agent.py, every call goes over REST with the serving endpoint's
injected credentials (WorkspaceClient().api_client), and this file imports nothing from its
siblings because MLflow imports it in isolation when validating a code-based model.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Callable

import mlflow
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

LLM_ENDPOINT = os.environ.get("MYGPT_SUPERVISOR_LLM", "mygpt-chat")
RAG_ENDPOINT = os.environ.get("MYGPT_RAG_ENDPOINT", "mygpt-rag-agent")
GENIE_SPACE_ID = os.environ.get("MYGPT_GENIE_SPACE_ID", "")
TOOLS_SCHEMA = os.environ.get("MYGPT_TOOLS_SCHEMA", "structural_biology.mygpt_tools")
CORPORA = [c for c in os.environ.get(
    "MYGPT_CORPORA", "biomed,kinase_literature,health_policies_fr,health_policies_es").split(",") if c]

MAX_STEPS = int(os.environ.get("MYGPT_SUPERVISOR_MAX_STEPS", "4"))
MAX_TOKENS = 1500
GENIE_TIMEOUT_S = 90.0
GENIE_POLL_S = 2.0
GENIE_MAX_ROWS = 50
TOOL_RESULT_CHARS = 6000
HISTORY_MESSAGES = 6
ROUTES = ("auto", "docs", "genie", "tools")
LIBRARY_ACTIONS = ("create_library", "add_document_from_url", "reingest")
GENIE_DONE = ("COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED")

SYSTEM_PROMPT = """You are the router of MyGPT, St. Jude's governed research assistant. Pick the tool(s) that answer the user's question:
- search_documents: anything about what the documents, papers or policies in the user's library say. It returns a grounded, cited answer with trust scores. This is the default for questions about content. Pass the user's question as `query` unchanged unless it refers to earlier turns; then rewrite it as a standalone question.
- query_structured_data: counts, lists and joins over structured data: the human kinome (kinase groups, families, UniProt ids, synonyms), which papers in the BlueSky kinase library study which kinase, and the catalogue of documents per corpus. It cannot read document text.
- PDB / PubMed tools: 3D protein structures in the RCSB Protein Data Bank, and PubMed paper metadata. Prefer search_pdb_by_gene for "structures of <protein>".
- list_corpora, list_corpus_documents, list_my_libraries: which collections and documents exist.
- propose_library_action: the user asks to create a library, add a document from a URL, or re-run ingestion. It only proposes; the user confirms in the app, so never say it is done.
Rules: prefer exactly one tool; combine tools only when the question needs several kinds of information. Never answer questions about the documents from your own knowledge. If no tool fits (a greeting, a clarification), answer briefly without tools. When combining results, keep the [n] citations search_documents returned and name PDB ids and PubMed ids explicitly."""


# =========================================================================== #
# TRANSPORT  (REST; tests inject a fake)
# =========================================================================== #
class Transport:
    """The three kinds of remote calls the Supervisor makes."""

    def __init__(self) -> None:
        self._ws = None

    def _workspace(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient

            self._ws = WorkspaceClient()
        return self._ws

    def chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: str = "auto") -> dict:
        body: dict[str, Any] = {"messages": messages, "max_tokens": MAX_TOKENS}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice
        resp = self._workspace().api_client.do("POST", f"/serving-endpoints/{LLM_ENDPOINT}/invocations", body=body)
        return ((resp.get("choices") or [{}])[0].get("message")) or {}

    def rag(self, body: dict) -> dict:
        return self._workspace().api_client.do("POST", f"/serving-endpoints/{RAG_ENDPOINT}/invocations", body=body)

    def mcp(self, path: str, method: str, params: dict | None = None) -> dict:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method}
        if params is not None:
            body["params"] = params
        resp = self._workspace().api_client.do("POST", path, body=body)
        if resp.get("error"):
            raise RuntimeError(f"MCP {method} failed: {resp['error']}")
        return resp.get("result") or {}


# =========================================================================== #
# TOOL SPECS
# =========================================================================== #
def _fn(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required or []},
    }}


def search_documents_spec(corpus_choice: bool) -> dict:
    props: dict[str, Any] = {
        "query": {"type": "string", "description": "The question to answer from the documents"},
        "documents": {"type": "array", "items": {"type": "string"},
                      "description": "Optional: restrict to these document file names"},
        "section": {"type": "string", "description": "Optional: restrict to one section, e.g. Methods"},
    }
    if corpus_choice:
        props["corpus"] = {"type": "string", "enum": CORPORA,
                           "description": "Optional: search only this collection"}
    return _fn("search_documents",
               "Answer a question from the documents in the user's library, with citations and trust scores.",
               props, ["query"])


QUERY_STRUCTURED_SPEC = _fn(
    "query_structured_data",
    "Ask the structured-data assistant (Genie) a question over the human kinome, the BlueSky kinase "
    "paper library and the document catalogue. Returns its answer, the SQL it ran and the result rows.",
    {"question": {"type": "string", "description": "A standalone natural-language question"}}, ["question"])

LIST_MY_LIBRARIES_SPEC = _fn(
    "list_my_libraries", "List the document libraries the user can see (id, name, kind, visibility).", {})

PROPOSE_ACTION_SPEC = _fn(
    "propose_library_action",
    "Propose a change to the user's upload libraries. The app asks the user to confirm before anything happens.",
    {
        "action": {"type": "string", "enum": list(LIBRARY_ACTIONS)},
        "library_name": {"type": "string", "description": "create_library: the new library's name"},
        "visibility": {"type": "string", "enum": ["private", "group", "public"],
                       "description": "create_library: who can see it (default private)"},
        "group_name": {"type": "string", "description": "create_library with visibility=group: the workspace group"},
        "library_id": {"type": "string", "description": "add_document_from_url / reingest: an existing library id"},
        "url": {"type": "string", "description": "add_document_from_url: https URL of a PDF"},
    },
    ["action"],
)


# =========================================================================== #
# RESULT DECODING
# =========================================================================== #
def message_text(content: Any) -> str:
    """Chat content -> text. Reasoning models (Claude Sonnet 5) can return a list of
    reasoning + text blocks, also on the turn after a tool result (Phase-5 spike)."""
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict) and p.get("type") in ("text", "output_text")).strip()
    return str(content or "").strip()


def responses_text(resp: dict) -> str:
    """Text of a ResponsesAgent response (the RAG endpoint)."""
    parts = []
    for item in resp.get("output") or []:
        for c in item.get("content") or []:
            if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                parts.append(c.get("text") or "")
    return "".join(parts).strip()


def mcp_payload(result: dict) -> Any:
    """structuredContent when present, else the JSON (or text) of the first content block."""
    if isinstance(result.get("structuredContent"), dict):
        return result["structuredContent"]
    for c in result.get("content") or []:
        if c.get("type") == "text":
            try:
                return json.loads(c.get("text") or "")
            except (TypeError, ValueError):
                return c.get("text")
    return None


def decode_function_result(payload: Any) -> Any:
    """Managed-MCP UC-function results come as {columns, rows}. Scalar functions return one
    `output` cell (our tools put JSON in it); table functions return row dicts."""
    if not isinstance(payload, dict) or "columns" not in payload:
        return payload
    cols, rows = payload.get("columns") or [], payload.get("rows") or []
    if cols == ["output"] and len(rows) == 1:
        cell = rows[0][0]
        if isinstance(cell, str):
            try:
                return json.loads(cell)
            except ValueError:
                return cell
        return cell
    out: Any = [dict(zip(cols, r)) for r in rows]
    if payload.get("is_truncated"):
        out = {"rows": out, "truncated": True}
    return out


def _cell(v: Any) -> Any:
    if isinstance(v, dict):  # proto-style {"string_value": "..."} / {"null_value": ...}
        return next(iter(v.values()), None) if "null_value" not in v else None
    return v


def decode_genie(payload: dict) -> dict:
    """Genie MCP message -> {status, text, sql, description, columns, rows, conversation_id}."""
    content = payload.get("content") or {}
    out: dict[str, Any] = {
        "status": payload.get("status"), "conversation_id": payload.get("conversationId"),
        "text": "\n".join(t for t in content.get("textAttachments") or [] if t),
        "sql": None, "description": None, "columns": [], "rows": [], "row_count": 0,
    }
    for q in content.get("queryAttachments") or []:
        out["sql"], out["description"] = q.get("query"), q.get("description")
        sr = q.get("statement_response") or {}
        if isinstance(sr, str):
            try:
                sr = json.loads(sr)
            except ValueError:
                sr = {}
        out["columns"] = [c.get("name") for c in ((sr.get("manifest") or {}).get("schema") or {}).get("columns") or []]
        data = (sr.get("result") or {}).get("data_array") or []
        rows = [[_cell(v) for v in (r.get("values") if isinstance(r, dict) else r)] for r in data]
        out["row_count"] = (sr.get("manifest") or {}).get("total_row_count", len(rows))
        out["rows"] = rows[:GENIE_MAX_ROWS]
        break
    return out


def _clip(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= TOOL_RESULT_CHARS else text[:TOOL_RESULT_CHARS] + " …[truncated]"


def _args(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"malformed tool arguments: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


# =========================================================================== #
# AGENT
# =========================================================================== #
class MyGPTSupervisor(ResponsesAgent):
    def __init__(self, transport: Transport | None = None, sleep: Callable[[float], None] = time.sleep) -> None:
        self.t = transport or Transport()
        self._sleep = sleep
        self._fn_tools: dict[str, tuple[str, dict]] | None = None  # short name -> (MCP name, spec)
        self._fn_tools_error: str | None = None

    # ------------------------------------------------------------------ tools
    @property
    def functions_path(self) -> str:
        return "/api/2.0/mcp/functions/" + TOOLS_SCHEMA.replace(".", "/")

    def function_tools(self) -> dict[str, tuple[str, dict]]:
        """UC functions in TOOLS_SCHEMA as chat tools, keyed by the short function name."""
        if self._fn_tools is None:
            try:
                listed = self.t.mcp(self.functions_path, "tools/list").get("tools") or []
                self._fn_tools = {}
                for tool in listed:
                    short = tool["name"].rsplit("__", 1)[-1]
                    self._fn_tools[short] = (tool["name"], _fn(
                        short, tool.get("description") or "",
                        (tool.get("inputSchema") or {}).get("properties") or {},
                        (tool.get("inputSchema") or {}).get("required") or []))
                self._fn_tools_error = None
            except Exception as exc:  # noqa: BLE001  (tools are optional; documents still work)
                self._fn_tools_error = f"{type(exc).__name__}: {exc}"
                print(f"[supervisor] function tools unavailable: {self._fn_tools_error}")
                return {}  # not cached: retry on the next request
        return self._fn_tools

    def tool_specs(self, route: str, ci: dict) -> list[dict]:
        specs: list[dict] = []
        if route == "auto":
            specs.append(search_documents_spec(corpus_choice=not ci.get("corpus") and ci.get("sources") is None))
        if route in ("auto", "genie") and GENIE_SPACE_ID:
            specs.append(QUERY_STRUCTURED_SPEC)
        if route in ("auto", "tools"):
            specs += [spec for _, spec in self.function_tools().values()]
            if ci.get("libraries") is not None:
                specs.append(LIST_MY_LIBRARIES_SPEC)
            if ci.get("allow_actions"):
                specs.append(PROPOSE_ACTION_SPEC)
        return specs

    # --------------------------------------------------------- tool execution
    def search_documents(self, args: dict, ci: dict, state: dict) -> dict:
        """Call the RAG agent inside the app's scope. The LLM can narrow corpus / documents /
        section, never widen them; upload-library hits (ci.sources) are forwarded as-is."""
        rag_ci: dict[str, Any] = {k: ci[k] for k in ("dataset", "model", "k", "instructions", "sources") if k in ci}
        rag_ci["include_context"] = True
        app_corpus = ci.get("corpus") or None
        # A corpus choice only narrows the all-governed-documents scope; upload libraries
        # arrive pre-retrieved (ci.sources), so there is nothing to choose.
        llm_corpus = args.get("corpus") if args.get("corpus") in CORPORA and "sources" not in ci else None
        corpus = app_corpus or llm_corpus
        if corpus:
            rag_ci["corpus"] = corpus
        app_focus = ci.get("focus") or {}
        app_docs = [d for d in app_focus.get("documents") or [] if isinstance(d, str)]
        llm_docs = [d for d in args.get("documents") or [] if isinstance(d, str)] if isinstance(args.get("documents"), list) else []
        docs = ([d for d in llm_docs if d in app_docs] or app_docs) if app_docs else llm_docs
        section = app_focus.get("section") or args.get("section")
        focus = {k: v for k, v in (("documents", docs), ("section", section)) if v}
        if focus:
            rag_ci["focus"] = focus
        query = str(args.get("query") or "").strip() or state["question"]

        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as span:
            span.set_inputs({"query": query, "corpus": corpus, "focus": focus, "provided": "sources" in rag_ci})
            resp = self.t.rag({"input": [{"role": "user", "content": query}], "custom_inputs": rag_ci})
            co = resp.get("custom_outputs") or {}
            sources = co.get("sources") or []
            # Same Document shape as the RAG agent's span, so RetrievalGroundedness reads the
            # Supervisor trace too (the RAG endpoint's own trace is separate).
            span.set_outputs([
                {"page_content": s.get("context") or s.get("snippet") or "",
                 "metadata": {"doc_uri": s.get("source_name"), "chunk_id": s.get("chunk_id"),
                              "corpus": s.get("corpus"), "section": s.get("section"), "page": s.get("page")}}
                for s in sources
            ])
        answer = responses_text(resp)
        co["sources"] = [{k: v for k, v in s.items() if k not in ("context", "chunk_id")} for s in sources]
        state["rag"].append({"answer": answer, "query": query, "custom_outputs": co})
        return {
            "answer": answer,
            "sources": [{"n": i + 1, "source_name": s.get("source_name"), "page": s.get("page"),
                         "section": s.get("section")} for i, s in enumerate(sources)],
            "trust": {"qrs": co.get("qrs"), "ars": co.get("ars"), "hallucination_index": co.get("hi_eq")},
            **({"error": co["retrieve_error"]} if co.get("retrieve_error") else {}),
        }

    def query_structured_data(self, args: dict, state: dict) -> dict:
        question = str(args.get("question") or "").strip() or state["question"]
        path = f"/api/2.0/mcp/genie/{GENIE_SPACE_ID}"
        msg = mcp_payload(self.t.mcp(path, "tools/call", {
            "name": f"query_space_{GENIE_SPACE_ID}", "arguments": {"query": question}})) or {}
        deadline = time.monotonic() + GENIE_TIMEOUT_S
        while (msg.get("status") or "") not in GENIE_DONE and time.monotonic() < deadline:
            self._sleep(GENIE_POLL_S)
            msg = mcp_payload(self.t.mcp(path, "tools/call", {
                "name": f"poll_response_{GENIE_SPACE_ID}",
                "arguments": {"conversation_id": msg.get("conversationId"), "message_id": msg.get("messageId")}})) or {}
        genie = decode_genie(msg)
        if genie["status"] not in GENIE_DONE:
            genie["status"] = "TIMEOUT"
        state["genie"] = genie
        return {k: genie[k] for k in ("status", "text", "sql", "columns", "row_count")} | {"rows": genie["rows"][:20]}

    def propose_library_action(self, args: dict, ci: dict, state: dict) -> dict:
        action = args.get("action")
        if action not in LIBRARY_ACTIONS:
            raise ValueError(f"unknown action {action!r}; expected one of {LIBRARY_ACTIONS}")
        proposal: dict[str, Any] = {"id": str(uuid.uuid4()), "action": action}
        if action == "create_library":
            name = str(args.get("library_name") or "").strip()
            if not name:
                raise ValueError("create_library needs library_name")
            proposal |= {"library_name": name[:100], "visibility": args.get("visibility") or "private"}
            if proposal["visibility"] == "group":
                proposal["group_name"] = str(args.get("group_name") or "")
        else:
            lib_id = str(args.get("library_id") or "")
            uploads = {str(lib.get("id")): lib for lib in ci.get("libraries") or [] if lib.get("kind") == "upload"}
            pending_new = [a for a in state["actions"] if a["action"] == "create_library"]
            if lib_id not in uploads and not pending_new:
                raise ValueError("library_id must be one of the user's upload libraries (see list_my_libraries)")
            proposal["library_id"] = lib_id if lib_id in uploads else None  # None = the library proposed above
            if action == "add_document_from_url":
                url = str(args.get("url") or "")
                if not url.startswith("https://"):
                    raise ValueError("url must be an https URL")
                proposal["url"] = url
        state["actions"].append(proposal)
        return {"status": "proposed", "action_id": proposal["id"],
                "note": "Nothing has happened yet: the user must confirm this in the app."}

    def run_tool(self, call: dict, ci: dict, state: dict) -> str:
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        started = time.monotonic()
        record: dict[str, Any] = {"name": name, "ok": True}
        with mlflow.start_span(name=name or "tool", span_type=SpanType.TOOL) as span:
            try:
                args = _args(fn.get("arguments"))
                record["args"] = args
                span.set_inputs(args)
                if name == "search_documents":
                    result: Any = self.search_documents(args, ci, state)
                    state["routes"].append("documents")
                elif name == "query_structured_data" and GENIE_SPACE_ID:
                    result = self.query_structured_data(args, state)
                    state["routes"].append("genie")
                elif name == "list_my_libraries" and ci.get("libraries") is not None:
                    result = [{k: lib.get(k) for k in ("id", "name", "kind", "visibility", "corpus")}
                              for lib in ci.get("libraries") or []][:50]
                    state["routes"].append("libraries")
                elif name == "propose_library_action" and ci.get("allow_actions"):
                    result = self.propose_library_action(args, ci, state)
                    state["routes"].append("libraries")
                elif name in self.function_tools():
                    mcp_name = self.function_tools()[name][0]
                    result = decode_function_result(mcp_payload(self.t.mcp(
                        self.functions_path, "tools/call", {"name": mcp_name, "arguments": args})))
                    state["routes"].append("tools")
                else:
                    raise ValueError(f"unknown tool {name!r}")
            except Exception as exc:  # noqa: BLE001  (reported to the LLM, which can recover)
                record["ok"] = False
                result = {"error": f"{type(exc).__name__}: {exc}"}
            text = _clip(result)
            span.set_outputs(text)
        record["latency_ms"] = int((time.monotonic() - started) * 1000)
        record["result"] = text[:300]
        state["tool_calls"].append(record)
        return text

    # ------------------------------------------------------------------ main
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        ci = dict(request.custom_inputs or {})
        route = ci.get("route") if ci.get("route") in ROUTES else "auto"
        turns = _conversation(request)
        question = turns[-1]["content"] if turns and turns[-1]["role"] == "user" else ""
        state: dict[str, Any] = {"question": question, "rag": [], "genie": None, "actions": [],
                                 "tool_calls": [], "routes": [], "error": None}
        multi_turn = len(turns) > 1

        answer: str | None = None
        verbatim = False
        if route == "docs":
            self.run_tool(_call("search_documents", {"query": question}), ci, state)
            verbatim = True
        else:
            messages: list[dict] = [{"role": "system", "content": _system_prompt(ci)}] + turns
            tools = self.tool_specs(route, ci)
            for step in range(MAX_STEPS):
                forced = step == 0 and route in ("genie", "tools") and bool(tools)
                try:
                    msg = self._chat(messages, tools, "required" if forced else "auto")
                except Exception as exc:  # noqa: BLE001
                    state["error"] = f"{type(exc).__name__}: {exc}"
                    if step == 0 and route == "auto":
                        # Router down: degrade to the pre-Phase-5 behaviour (documents only).
                        self.run_tool(_call("search_documents", {"query": question}), ci, state)
                        verbatim = True
                    break
                calls = msg.get("tool_calls") or []
                if not calls:
                    answer = message_text(msg.get("content"))
                    break
                if step == 0 and len(calls) == 1 and (calls[0].get("function") or {}).get("name") == "search_documents":
                    call = calls[0]
                    if not multi_turn:
                        # Single-turn: the RAG agent sees the user's exact question (eval parity).
                        args = _safe_args(call)
                        call = _call("search_documents", {**args, "query": question})
                    self.run_tool(call, ci, state)
                    verbatim = bool(state["rag"])
                    break
                messages.append({"role": "assistant", "content": message_text(msg.get("content")) or None,
                                 "tool_calls": calls})
                for call in calls:
                    messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": self.run_tool(call, ci, state)})
            else:
                # Step budget spent: one last turn without tools to write the answer.
                try:
                    answer = message_text(self._chat(messages, None).get("content"))
                except Exception as exc:  # noqa: BLE001
                    state["error"] = f"{type(exc).__name__}: {exc}"

        if verbatim and state["rag"]:
            answer = state["rag"][0]["answer"]
        if not answer:
            answer = "Sorry, I couldn't produce an answer." + (f" ({state['error']})" if state["error"] else "")
        return self._response(answer, verbatim, ci, state)

    @mlflow.trace(span_type=SpanType.CHAT_MODEL)
    def _chat(self, messages: list[dict], tools: list[dict] | None, tool_choice: str = "auto") -> dict:
        return self.t.chat(messages, tools, tool_choice)

    def _response(self, answer: str, verbatim: bool, ci: dict, state: dict) -> ResponsesAgentResponse:
        routes = list(dict.fromkeys(state["routes"])) or ["direct"]
        rag = state["rag"][0]["custom_outputs"] if state["rag"] else {}
        trace_id = _active_trace_id()
        try:
            mlflow.update_current_trace(tags={
                "mygpt.agent": "supervisor", "mygpt.route": ",".join(routes),
                "mygpt.corpus": ci.get("corpus") or "all", "mygpt.model": LLM_ENDPOINT,
            })
        except Exception:  # noqa: BLE001  (no active trace, e.g. tracing disabled)
            pass
        outputs = {
            "route": routes,
            "scores_scope": ("answer" if verbatim else "documents_part") if rag else None,
            "model": rag.get("model") or LLM_ENDPOINT,
            "corpus": rag.get("corpus", ci.get("corpus")),
            "qrs": rag.get("qrs"), "ars": rag.get("ars"), "hi_eq": rag.get("hi_eq"), "hi_rf": rag.get("hi_rf"),
            "sources": rag.get("sources") or [],
            "retrieve_error": rag.get("retrieve_error"),
            # Feedback lands on the RAG trace when there is one (it holds the retrieval), else here.
            "trace_id": rag.get("trace_id") or trace_id,
            "supervisor_trace_id": trace_id,
            "supervisor_model": LLM_ENDPOINT,
            "tool_calls": state["tool_calls"],
            "genie": state["genie"],
            "actions": state["actions"],
            "error": state["error"] or self._fn_tools_error,
        }
        item = self.create_text_output_item(answer, str(uuid.uuid4()))
        return ResponsesAgentResponse(output=[item], custom_outputs=outputs)


# =========================================================================== #
# HELPERS
# =========================================================================== #
def _call(name: str, args: dict) -> dict:
    return {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _safe_args(call: dict) -> dict:
    try:
        return _args((call.get("function") or {}).get("arguments"))
    except ValueError:
        return {}


def _system_prompt(ci: dict) -> str:
    lib = ci.get("library") or {}
    scope = f"The user is working in the library {lib.get('name')!r} ({lib.get('kind') or 'corpus'})." if lib.get("name") else ""
    if ci.get("corpus"):
        scope += f" Document search is limited to the {ci['corpus']} collection."
    return SYSTEM_PROMPT + (f"\n{scope.strip()}" if scope else "")


def _conversation(request: ResponsesAgentRequest) -> list[dict]:
    """User / assistant text turns of the request (the app sends recent history as input
    messages), last HISTORY_MESSAGES kept, ending with the current question."""
    turns = []
    for msg in request.input:
        d = msg if isinstance(msg, dict) else msg.model_dump()
        role = d.get("role")
        if role not in ("user", "assistant"):
            continue
        content = d.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = str(content or "").strip()
        if text:
            turns.append({"role": role, "content": text})
    return turns[-HISTORY_MESSAGES:]


def _active_trace_id() -> str | None:
    try:
        return mlflow.get_active_trace_id()
    except Exception:  # noqa: BLE001
        return None


mlflow.models.set_model(MyGPTSupervisor())
