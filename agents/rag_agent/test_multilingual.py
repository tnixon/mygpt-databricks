"""Tests for the Phase-6 multilingual path (translation, per-language reranker, answer BM25)
with a fake workspace transport and fake cross-encoders — no torch, no endpoints.

    uv run --no-project --with-requirements eval/requirements.txt pytest -q agents/rag_agent/test_multilingual.py
"""

from __future__ import annotations

import dataclasses
import json
import os
from unittest.mock import MagicMock

import langsmith
import mlflow
import numpy as np
from mlflow.types.responses import ResponsesAgentRequest

import agent as ag

mlflow.tracing.disable()
os.environ["LANGSMITH_TRACING"] = "false"  # a developer's .env must never send test traces

FR_CHUNK = "La politique nationale prévoit un dépistage du cancer chez l'enfant."
EN_CHUNK = "Imatinib binds the ABL1 kinase domain."


class FakeApi:
    """Routes api_client.do(...) by path; records translate / chat / search calls."""

    def __init__(self, rows: list[list], fail_translate: bool = False):
        self.rows = rows
        self.fail_translate = fail_translate
        self.calls: list[tuple[str, dict]] = []

    def do(self, method, path, body=None):
        self.calls.append((path, json.loads(json.dumps(body))))
        if path.endswith("/vector-search/indexes/" + ag.VS_INDEX + "/query"):
            cols = ["chunk_id", "source_name", "corpus", "section", "page", "chunk_text", "score"]
            return {"manifest": {"columns": [{"name": c} for c in cols]}, "result": {"data_array": self.rows}}
        if f"/serving-endpoints/{ag.TRANSLATE_ENDPOINT}/" in path:
            if self.fail_translate:
                raise RuntimeError("gateway down")
            text = body["messages"][-1]["content"]
            return {"choices": [{"message": {"content": f"FR<{text}>"}}]}
        if "/serving-endpoints/" in path:  # answer model
            return {"choices": [{"message": {"content": "Yes. The policy covers childhood cancer [1]."}}]}
        raise AssertionError(path)

    def paths(self, needle: str) -> list[dict]:
        return [b for p, b in self.calls if needle in p]


class FakeCrossEncoder:
    def __init__(self, name):
        self.name = name

    def predict(self, pairs):
        if self.name == ag.NLI_MODEL:
            return np.array([[0.1, 0.8, 0.1]] * len(pairs))
        return np.array([0.9] * len(pairs))


def make_agent(api: FakeApi) -> ag.MyGPTRagAgent:
    a = ag.MyGPTRagAgent()
    a._ws = type("W", (), {"api_client": api})()
    a._reranker_model = lambda name=ag.RERANK_MODEL: FakeCrossEncoder(name)
    a._nli_model = lambda: FakeCrossEncoder(ag.NLI_MODEL)
    a._embed = lambda texts: [[1.0, 0.0]] + [[0.9, 0.1]] * (len(texts) - 1)
    return a


def ask(a, question: str, **ci) -> dict:
    req = ResponsesAgentRequest(input=[{"role": "user", "content": question}], custom_inputs=ci)
    resp = a.predict(req)
    text = resp.output[0].content[0]["text"]
    return {"answer": text, **resp.custom_outputs}


def fr_rows():
    return [["c1", "plan.pdf", "health_policies_fr", "Objectifs", 3, FR_CHUNK, 0.8]]


def test_packaged_config_is_language_driven():
    fr = ag._config_fallback("health_policies_fr")
    assert (fr.documents_language, fr.reranker, fr.reranker_cutoff, fr.translate_query) == (
        "french", ag.RERANK_MODEL_MULTILINGUAL, 0.05, True)
    # answer translation + answer-side BM25 are opt-in (latency / uncalibrated BM25 window)
    assert not fr.translate_answer and not fr.answer_bm25
    en = ag._config_fallback("kinase_literature")
    assert (en.reranker, en.reranker_cutoff, en.translate_query) == (ag.RERANK_MODEL, 0.10, False)
    assert ag._config_fallback("default_german").documents_language == "german"
    unk = ag._config_fallback("default_unknown")
    assert unk.reranker == ag.RERANK_MODEL_MULTILINGUAL and not unk.translate_query


def test_english_question_on_french_corpus_translates_the_query_only():
    api = FakeApi(fr_rows())
    out = ask(make_agent(api), "Does the policy mention children?", corpus="health_policies_fr")
    # the translated query drives retrieval (and rerank)...
    assert api.paths("/vector-search/")[0]["query_text"] == "FR<Does the policy mention children?>"
    assert out["translated_query"] == "FR<Does the policy mention children?>"
    # ...the model sees the original question; by default the answer isn't translated.
    chat = [b for p, b in api.calls if "/serving-endpoints/" in p and ag.TRANSLATE_ENDPOINT not in p][0]
    assert "Question: Does the policy mention children?" in chat["messages"][0]["content"]
    assert out["answer"] == "Yes. The policy covers childhood cancer [1]."
    assert out["translated_answer"] is None and len(api.paths(ag.TRANSLATE_ENDPOINT)) == 1
    assert out["documents_language"] == "french" and out["translate_error"] is None
    assert out["reranker"] == ag.RERANK_MODEL_MULTILINGUAL
    src = out["sources"][0]
    assert src["rerank_score"] == 0.9 and src["reranker"] == ag.RERANK_MODEL_MULTILINGUAL
    assert src["nli"] is None  # NLI is English-only
    assert src["answer_bm25_score"] is None  # answer_bm25 off by default


def test_translate_answer_and_answer_bm25_opt_in():
    api = FakeApi(fr_rows())
    a = make_agent(api)
    a._cfg_cache["health_policies_fr"] = dataclasses.replace(ag._config_fallback("health_policies_fr"), answer_bm25=True)
    out = ask(a, "Does the policy mention children?", corpus="health_policies_fr", translate_answer=True)
    # the answer is translated for scoring only (MyGPT); the user still gets the original
    assert out["answer"] == "Yes. The policy covers childhood cancer [1]."
    assert out["translated_answer"] == "FR<Yes. The policy covers childhood cancer [1].>"
    assert out["sources"][0]["answer_bm25_score"] is not None


def test_translate_false_and_same_language_skip_translation():
    api = FakeApi(fr_rows())
    out = ask(make_agent(api), "Does the policy mention children?", corpus="health_policies_fr", translate=False)
    assert out["translated_query"] is None and out["translated_answer"] is None
    api2 = FakeApi(fr_rows())
    out2 = ask(make_agent(api2), "La politique mentionne-t-elle les enfants ?", corpus="health_policies_fr",
               query_language="french")
    assert out2["translated_query"] is None
    assert not api.paths(ag.TRANSLATE_ENDPOINT) and not api2.paths(ag.TRANSLATE_ENDPOINT)


def test_translation_failure_degrades_to_original_text():
    api = FakeApi(fr_rows(), fail_translate=True)
    out = ask(make_agent(api), "Does the policy mention children?", corpus="health_policies_fr")
    assert api.paths("/vector-search/")[0]["query_text"] == "Does the policy mention children?"
    assert "query: RuntimeError: gateway down" in out["translate_error"] and out["translated_query"] is None
    out2 = ask(make_agent(FakeApi(fr_rows(), fail_translate=True)), "Does the policy mention children?",
               corpus="health_policies_fr", translate_answer=True)
    assert "answer: RuntimeError" in out2["translate_error"] and out2["translated_answer"] is None
    assert out["answer"].startswith("Yes.") and out["qrs"] is not None


def test_english_corpus_unchanged_no_translation_and_nli():
    api = FakeApi([["k1", "p.pdf", "kinase_literature", "Results", 2, EN_CHUNK, 0.8]])
    out = ask(make_agent(api), "What binds ABL1?", corpus="kinase_literature")
    assert not api.paths(ag.TRANSLATE_ENDPOINT)
    assert out["sources"][0]["reranker"] == ag.RERANK_MODEL and out["sources"][0]["nli"] == "entailment"


def test_mixed_scope_reranks_each_source_by_corpus_language():
    rows = [["k1", "p.pdf", "kinase_literature", "Results", 2, EN_CHUNK, 0.8],
            ["c1", "plan.pdf", "health_policies_fr", "Objectifs", 3, FR_CHUNK, 0.7]]
    api = FakeApi(rows)
    out = ask(make_agent(api), "Childhood cancer screening?", dataset="mygpt")
    by_corpus = {s["corpus"]: s["reranker"] for s in out["sources"]}
    assert by_corpus == {"kinase_literature": ag.RERANK_MODEL, "health_policies_fr": ag.RERANK_MODEL_MULTILINGUAL}
    assert not api.paths(ag.TRANSLATE_ENDPOINT)  # no single document language -> no translation


def test_upload_library_sources_follow_the_library_config():
    api = FakeApi([])
    sources = [{"chunk_id": "u1", "source_name": "doc.pdf", "corpus": "upload:abc", "chunk_text": FR_CHUNK,
                "similarity": 0.8, "bm25_score": 0.4}]
    out = ask(make_agent(api), "Does the policy mention children?", dataset="health_policies_fr", sources=sources)
    assert out["translated_query"].startswith("FR<")
    assert out["sources"][0]["reranker"] == ag.RERANK_MODEL_MULTILINGUAL
    assert not api.paths("/vector-search/")


# --------------------------------------------------------------------------- LangSmith
def test_langsmith_off_by_default():
    api = FakeApi([["k1", "p.pdf", "kinase_literature", "Results", 2, EN_CHUNK, 0.8]])
    assert ask(make_agent(api), "What binds ABL1?", corpus="kinase_literature")["langsmith_run_id"] is None


def test_langsmith_mirrors_the_pipeline_under_the_supervisors_run():
    client = MagicMock()
    api = FakeApi(fr_rows())
    with langsmith.tracing_context(enabled=True, client=client, project_name="test"):
        with langsmith.trace("retrieve", "retriever") as supervisor_run:
            headers = supervisor_run.to_headers()
        out = ask(make_agent(api), "Does the policy mention children?", corpus="health_policies_fr",
                  langsmith_parent=headers)
    runs = {c.kwargs["name"]: c.kwargs for c in client.create_run.call_args_list}
    assert {"mygpt-rag-agent", "translate", "retrieve", "_rerank", "_score_question", "_generate",
            "_score_answer"} <= set(runs)
    root = runs["mygpt-rag-agent"]
    assert root["parent_run_id"] == supervisor_run.id and root["trace_id"] == supervisor_run.trace_id
    assert "langsmith_parent" not in root["inputs"]["custom_inputs"]
    assert out["langsmith_run_id"] == str(root["id"])
    # metadata carries labels only (it isn't redacted by LANGSMITH_HIDE_*)
    update = next(c.kwargs for c in client.update_run.call_args_list if c.kwargs["run_id"] == root["id"])
    meta = update["extra"]["metadata"]
    assert (meta["corpus"], meta["retrieval"], meta["language"], meta["translated"]) == (
        "health_policies_fr", "ai_search", "french", "true")
    assert update["outputs"]["answer"] == out["answer"] and update["outputs"]["qrs"] == out["qrs"]
