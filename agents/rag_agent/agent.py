"""MyGPT retrieval + scoring + generation as an MLflow ResponsesAgent.

Deployed to its own Model Serving endpoint (Mosaic AI Agent Framework). It owns the
full MyGPT-grade pipeline and returns the answer plus the QRS / ARS / Hallucination
-Index trust metrics and per-source scores via ``custom_outputs``:

  1. retrieve   AI Search (ANN + hybrid) with document/section focus filters
  2. rerank     cross-encoder/qnli-electra-base (query <-> source relevance)
  3. QRS        semantic (per-model distance calibration) + BM25 + rerank
  4. generate   FMAPI chat (Claude / Llama / GPT-OSS)
  5. NLI        cross-encoder/nli-deberta-v3-base (answer <-> source entailment)
  6. ARS + HI   answer relevance + hallucination index (equation; RF toggle)

Fidelity = Balanced (see PLAN.md): the BM25 term is derived from AI Search hybrid
scores rescored per source rather than a separate full-corpus bm25s index.

Request contract (from the app BFF):
  input:          [{role: "user", content: "<question>"}, ...]
  custom_inputs:  {dataset?: str, model?: "sonnet"|"llama"|"gptoss",
                   focus?: {documents?: [str], section?: str}, k?: int}

Heavy deps (sentence-transformers/torch, databricks-vectorsearch) resolve on the
GPU serving image; this module is import-light where possible so load failures are
localized and logged rather than crashing the endpoint.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

import hi_random_forest
import scoring
from config import DatasetConfig, load_dataset_config

VS_INDEX = os.environ.get("MYGPT_VS_INDEX", "structural_biology.mygpt.doc_chunks_index")
VS_ENDPOINT = os.environ.get("MYGPT_VS_ENDPOINT", "mygpt-vs")
EMBED_ENDPOINT = os.environ.get("MYGPT_EMBED_ENDPOINT", "mygpt-bge-m3")

MODEL_ALIASES = {
    "sonnet": os.environ.get("MYGPT_LLM_SONNET", "databricks-claude-sonnet-5"),
    "llama": os.environ.get("MYGPT_LLM_LLAMA", "databricks-llama-4-maverick"),
    "gptoss": os.environ.get("MYGPT_LLM_GPTOSS", "databricks-gpt-oss-120b"),
}

RERANK_MODEL = "cross-encoder/qnli-electra-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


class MyGPTRagAgent(ResponsesAgent):
    def __init__(self) -> None:
        self._cfg_cache: dict[str, DatasetConfig] = {}
        self._reranker = None
        self._nli = None
        self._vs_index = None
        self._llm = None

    # --- lazy clients ------------------------------------------------------- #
    def _reranker_model(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(RERANK_MODEL, activation_fn=torch.nn.Sigmoid())
        return self._reranker

    def _nli_model(self):
        if self._nli is None:
            from sentence_transformers import CrossEncoder

            self._nli = CrossEncoder(NLI_MODEL)
        return self._nli

    def _index(self):
        if self._vs_index is None:
            from databricks.vectorsearch.client import VectorSearchClient

            self._vs_index = VectorSearchClient(disable_notice=True).get_index(
                endpoint_name=VS_ENDPOINT, index_name=VS_INDEX
            )
        return self._vs_index

    def _llm_client(self):
        if self._llm is None:
            from databricks.sdk import WorkspaceClient

            self._llm = WorkspaceClient().serving_endpoints.get_open_ai_client()
        return self._llm

    def _config(self, dataset: str) -> DatasetConfig:
        if dataset not in self._cfg_cache:
            self._cfg_cache[dataset] = load_dataset_config(dataset)
        return self._cfg_cache[dataset]

    # --- embeddings (bge-m3) ----------------------------------------------- #
    def _embed(self, texts: list[str]) -> list[list[float]]:
        client = self._llm_client()
        resp = client.embeddings.create(model=EMBED_ENDPOINT, input=texts)
        return [d.embedding for d in resp.data]

    # --- pipeline ----------------------------------------------------------- #
    @mlflow.trace(span_type="RETRIEVER")
    def _retrieve(self, query: str, cfg: DatasetConfig, focus: dict, k: int) -> list[dict]:
        """AI Search hybrid query with optional document/section focus filters."""
        filters: dict[str, Any] = {}
        docs = focus.get("documents") or []
        section = focus.get("section")
        if docs:
            filters["source_name"] = docs  # list -> OR membership
        if section and section != "all":
            filters["section"] = section

        res = self._index().similarity_search(
            query_text=query,
            columns=["chunk_id", "source_name", "section", "page", "chunk_text"],
            num_results=k,
            query_type="HYBRID",
            filters=filters or None,
        )
        cols = [c["name"] for c in res["manifest"]["columns"]]
        rows = res.get("result", {}).get("data_array", []) or []
        sources: list[dict] = []
        for rank, row in enumerate(rows, start=1):
            r = dict(zip(cols, row))
            sim = float(r.get("score", 0.0))  # AI Search hybrid similarity (higher better)
            sources.append(
                {
                    "chunk_id": r.get("chunk_id"),
                    "source_name": r.get("source_name") or "",
                    "section": r.get("section") or "",
                    "page": None if r.get("page") is None else int(r["page"]),
                    "context": str(r.get("chunk_text") or ""),
                    "similarity": sim,
                    "vector_distance_raw": round(1.0 - sim, 4),  # cosine distance space
                    "bm25_score": round(sim, 4),  # Balanced: hybrid score as the keyword proxy
                    "rank": rank,
                }
            )
        return sources

    @mlflow.trace(span_type="RERANKER")
    def _rerank(self, query: str, sources: list[dict], cfg: DatasetConfig) -> list[dict]:
        if not sources or not cfg.use_reranker:
            for s in sources:
                s["rerank_score"] = 0.0
            return sources
        model = self._reranker_model()
        scores = model.predict([[query, s["context"]] for s in sources])
        kept = []
        for s, sc in zip(sources, scores):
            s["rerank_score"] = round(float(sc), 3)
            if s["rerank_score"] > cfg.reranker_cutoff:
                kept.append(s)
        kept.sort(key=lambda s: s["rerank_score"], reverse=True)
        for i, s in enumerate(kept, start=1):
            s["rank"] = i
        return kept or sources  # never return empty — fall back to the unfiltered set

    @mlflow.trace(span_type="PARSER")
    def _score_question(self, sources: list[dict], cfg: DatasetConfig) -> tuple[float, float, list[dict]]:
        distances = [s["vector_distance_raw"] for s in sources]
        semantic, normalized = scoring.question_semantic_score(distances, cfg.cutoffs)
        for s, nd in zip(sources, normalized):
            s["vector_score"] = round(nd, 3)
        keyword = (sum(s["bm25_score"] for s in sources) / len(sources)) if (sources and cfg.use_bm25) else 0.0
        rerank = (sum(s["rerank_score"] for s in sources) / len(sources)) if sources else 0.0
        qrs = scoring.question_relevance_score(semantic, keyword, rerank, cfg.scoring)
        return qrs, semantic, sources

    @mlflow.trace(span_type="LLM")
    def _generate(self, query: str, sources: list[dict], model_alias: str) -> str:
        context = "\n\n".join(
            f"[{i + 1}] {s['source_name']} (p{s['page'] if s['page'] is not None else '?'}):\n{s['context']}"
            for i, s in enumerate(sources)
        )
        prompt = (
            "You are MyGPT, a governed research assistant for St. Jude. Answer the question "
            "using ONLY the numbered sources below, and cite them inline as [n]. If the sources "
            "do not contain the answer, say you couldn't find it in the documents.\n\n"
            f"Sources:\n{context}\n\nQuestion: {query}"
        )
        endpoint = MODEL_ALIASES.get(model_alias, MODEL_ALIASES["sonnet"])
        resp = self._llm_client().chat.completions.create(
            model=endpoint, messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content or ""

    @mlflow.trace(span_type="PARSER")
    def _score_answer(
        self, answer: str, sources: list[dict], question_semantic: float, cfg: DatasetConfig
    ) -> dict:
        if not answer or not sources:
            return {"ars": 0.0, "hi_eq": 100.0, "hi_rf": None, "nli": []}

        # answer-vs-context distances via bge-m3 (cosine distance = 1 - cos sim)
        vectors = self._embed([answer] + [s["context"] for s in sources])
        a_vec, s_vecs = vectors[0], vectors[1:]
        answer_distances = [round(1.0 - _cosine(a_vec, sv), 4) for sv in s_vecs]
        answer_semantic, answer_vec_scores = scoring.answer_vector_scores(answer_distances, cfg.cutoffs)

        # NLI entailment labels (answer -> each source)
        nli_labels: list[str] = []
        if cfg.documents_language == "english":
            model = self._nli_model()
            label_map = ["contradiction", "entailment", "neutral"]
            preds = model.predict([(answer, s["context"]) for s in sources])
            nli_labels = [label_map[int(p.argmax())] for p in preds]

        rerank_mean_100 = (sum(s["rerank_score"] for s in sources) / len(sources)) * 100
        answer_keyword_100 = (sum(s["bm25_score"] for s in sources) / len(sources)) * 100 if cfg.use_bm25 else 0.0
        ars = scoring.answer_relevance_score(answer_semantic, answer_keyword_100, rerank_mean_100, cfg.scoring)

        # HI (equation) — MyGPT drives this from the question SEMANTIC score
        hi_eq = scoring.hallucination_index_equation(question_semantic, ars, cfg.scoring)

        # HI (random forest) — best-effort toggle
        hi_rf = hi_random_forest.predict_hallucination_index(
            vector_scores_answer=answer_vec_scores,
            bm25_scores_answer=[s["bm25_score"] for s in sources],
            sources=sources,
            rerank_sentiments=[[lbl] for lbl in nli_labels],
        )
        return {"ars": ars, "hi_eq": hi_eq, "hi_rf": hi_rf, "nli": nli_labels}

    # --- ResponsesAgent entrypoint ----------------------------------------- #
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        ci = request.custom_inputs or {}
        dataset = ci.get("dataset", "mygpt")
        model_alias = ci.get("model", "sonnet")
        focus = ci.get("focus", {}) or {}
        k = int(ci.get("k", 5))
        query = _last_user_text(request)
        cfg = self._config(dataset)

        sources = self._retrieve(query, cfg, focus, k)
        if not sources:
            item = self.create_text_output_item(
                "I couldn't find anything relevant in the indexed documents.", str(uuid.uuid4())
            )
            return ResponsesAgentResponse(
                output=[item], custom_outputs={"qrs": 0, "ars": 0, "hi_eq": 100, "sources": []}
            )

        sources = self._rerank(query, sources, cfg)
        qrs, question_semantic, sources = self._score_question(sources, cfg)
        answer = self._generate(query, sources, model_alias)
        ans_scores = self._score_answer(answer, sources, question_semantic, cfg)
        for s, label in zip(sources, ans_scores["nli"]):
            s["nli"] = label

        item = self.create_text_output_item(answer, str(uuid.uuid4()))
        return ResponsesAgentResponse(
            output=[item],
            custom_outputs={
                "model": MODEL_ALIASES.get(model_alias, MODEL_ALIASES["sonnet"]),
                "qrs": qrs,
                "ars": ans_scores["ars"],
                "hi_eq": ans_scores["hi_eq"],
                "hi_rf": ans_scores["hi_rf"],
                "sources": [
                    {
                        "source_name": s["source_name"],
                        "section": s["section"],
                        "page": s["page"],
                        "snippet": s["context"][:240],
                        "vector_score": s.get("vector_score"),
                        "bm25_score": s.get("bm25_score"),
                        "rerank_score": s.get("rerank_score"),
                        "nli": s.get("nli"),
                    }
                    for s in sources
                ],
            },
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _last_user_text(request: ResponsesAgentRequest) -> str:
    for msg in reversed(request.input):
        d = msg if isinstance(msg, dict) else msg.model_dump()
        if d.get("role") == "user":
            content = d.get("content", "")
            if isinstance(content, list):  # responses-style content parts
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
    return ""


mlflow.models.set_model(MyGPTRagAgent())
