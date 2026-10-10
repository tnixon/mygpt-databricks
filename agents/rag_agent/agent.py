"""MyGPT retrieval + scoring + generation as a self-contained MLflow ResponsesAgent.

Deployed to its own Model Serving endpoint (Mosaic AI Agent Framework). Owns the
full MyGPT-grade pipeline and returns the answer plus QRS / ARS / Hallucination-Index
trust metrics and per-source scores via ``custom_outputs``:

  1. retrieve   AI Search (ANN + hybrid) with document/section focus filters
  2. rerank     cross-encoder/qnli-electra-base (query <-> source relevance)
  3. QRS        semantic (per-model distance calibration) + BM25 + rerank
  4. generate   FMAPI chat (Claude / Llama / GPT-OSS)
  5. NLI        cross-encoder/nli-deberta-v3-base (answer <-> source entailment)
  6. ARS + HI   answer relevance + hallucination index (equation; RF toggle)

The scoring math, config loader, and random-forest HI are INLINED here (rather than
imported from scoring.py / config.py / hi_random_forest.py) because MLflow imports
this file in isolation when validating a code-based model, so sibling imports are
not resolvable at log time. ``scoring.py`` + ``test_scoring.py`` remain the verified
reference these inlined functions mirror — keep them in sync.

Fidelity = Balanced (see PLAN.md): the BM25 term is derived from AI Search hybrid
scores rescored per source rather than a separate full-corpus bm25s index.
"""

from __future__ import annotations

import json
import os
import pickle
import uuid
from dataclasses import dataclass
from typing import Any, Sequence

import mlflow
import numpy as np
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

VS_INDEX = os.environ.get("MYGPT_VS_INDEX", "structural_biology.mygpt.doc_chunks_corpus_index")
VS_ENDPOINT = os.environ.get("MYGPT_VS_ENDPOINT", "mygpt-vs")
EMBED_ENDPOINT = os.environ.get("MYGPT_EMBED_ENDPOINT", "mygpt-bge-m3")
CATALOG_SCHEMA = os.environ.get("MYGPT_SCHEMA", "structural_biology.mygpt")

MODEL_ALIASES = {
    "sonnet": os.environ.get("MYGPT_LLM_SONNET", "databricks-claude-sonnet-5"),
    "llama": os.environ.get("MYGPT_LLM_LLAMA", "databricks-llama-4-maverick"),
    "gptoss": os.environ.get("MYGPT_LLM_GPTOSS", "databricks-gpt-oss-120b"),
}
RERANK_MODEL = "cross-encoder/qnli-electra-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


# =========================================================================== #
# SCORING  (mirror of scoring.py — verified by test_scoring.py)
# =========================================================================== #
@dataclass(frozen=True)
class ScoringConfig:
    a_qsem: float = 4.0
    b_qkey: float = -4.0
    c_qrank: float = -1.0
    qrs_raw_max: float = 3.03
    qrs_raw_min: float = -5.5
    x_asem: float = 5.0
    y_akey: float = -2.0
    z_arank: float = 0.0
    ars_raw_max: float = 3.65
    ars_raw_min: float = -6.42
    p_qrs: float = 1.0
    q_ars: float = 2.0
    hi_max: float = 0.8
    hi_min: float = 0.2
    bm25_best: float = 20.0
    bm25_worst: float = 0.0


@dataclass(frozen=True)
class EmbeddingCutoffs:
    best_q: float
    worst_q: float
    best_ac: float
    worst_ac: float


def min_max_normalization(data: Sequence[float], best_val: float, worst_val: float, reverse: bool = False) -> list[float]:
    out: list[float] = []
    span = (worst_val - best_val) if reverse else (best_val - worst_val)
    for value in data:
        out.append((worst_val - value) / span if reverse else (value - worst_val) / span)
    return out


def _buffered_window(best: float, worst: float, lo_frac: float, hi_frac: float) -> tuple[float, float]:
    buffer = (worst - best) * lo_frac
    return best - buffer, worst + (worst - best) * hi_frac


def question_semantic_score(distances: Sequence[float], cutoffs: EmbeddingCutoffs) -> tuple[float, list[float]]:
    best, worst = _buffered_window(cutoffs.best_q, cutoffs.worst_q, 0.1, 0.3)
    if len(distances) == 0:
        return 0.0, []
    if min(distances) > worst:
        semantic = 0.0
    else:
        mean_distance = sum(distances) / len(distances)
        semantic = round((1 - (mean_distance - best) / (worst - best)) * 100, 0)
    normalized = min_max_normalization(distances, best, worst, reverse=True)
    return semantic, normalized


def question_relevance_score(semantic_score: float, keyword_score: float, rerank_score: float, cfg: ScoringConfig) -> float:
    raw = round((semantic_score / 100 * cfg.a_qsem) + (keyword_score * cfg.b_qkey) + (rerank_score * cfg.c_qrank), 2)
    qrs = round(min_max_normalization([raw], cfg.qrs_raw_max, cfg.qrs_raw_min, False)[0] * 100, 0)
    return _clamp_0_100(qrs)


def answer_vector_scores(answer_distances: Sequence[float], cutoffs: EmbeddingCutoffs) -> tuple[float, list[float]]:
    best, worst = cutoffs.best_ac, cutoffs.worst_ac
    buffer = 0.05 * (worst - best)
    best, worst = best - buffer, worst + buffer
    if len(answer_distances) == 0:
        return 0.0, []
    vector_scores = [round(s, 3) for s in min_max_normalization(answer_distances, best, worst, reverse=False)]
    semantic = round(sum(vector_scores) / len(vector_scores) * 100, 2)
    return semantic, vector_scores


def answer_relevance_score(answer_semantic_score: float, answer_keyword_score: float, rerank_score: float, cfg: ScoringConfig) -> float:
    raw = round((answer_semantic_score / 100 * cfg.x_asem) + (answer_keyword_score / 100 * cfg.y_akey) + (rerank_score / 100 * cfg.z_arank), 2)
    ars = round(min_max_normalization([raw], cfg.ars_raw_max, cfg.ars_raw_min, False)[0] * 100, 0)
    return _clamp_0_100(ars)


def hallucination_index_equation(question_score: float, answer_relevance: float, cfg: ScoringConfig) -> float:
    if question_score == 0:
        hi_raw = 1.0
    else:
        denom = cfg.p_qrs + cfg.q_ars
        hi_raw = 1 - (cfg.p_qrs * (question_score / 100)) / denom - (cfg.q_ars * (answer_relevance / 100)) / denom
    return _clamp_0_100(round((hi_raw - cfg.hi_min) / (cfg.hi_max - cfg.hi_min) * 100, 0))


def _clamp_0_100(value: float) -> float:
    return 0.0 if value < 0 else 100.0 if value > 100 else float(value)


# =========================================================================== #
# RANDOM-FOREST HI  (mirror of hi_random_forest.py — best-effort toggle)
# =========================================================================== #
_RF_ARTIFACT: dict[str, Any] | None = None
_RF_ERROR: str | None = None


def _rf_artifact() -> dict[str, Any] | None:
    global _RF_ARTIFACT, _RF_ERROR
    if _RF_ARTIFACT is not None or _RF_ERROR is not None:
        return _RF_ARTIFACT
    path = os.environ.get("MYGPT_RF_MODEL_PATH", "")
    if path and os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                _RF_ARTIFACT = pickle.load(fh)
            return _RF_ARTIFACT
        except Exception as exc:  # noqa: BLE001
            _RF_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    _RF_ERROR = "rf model path missing"
    return None


def _safe_stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.max(arr)), float(np.min(arr)), float(np.std(arr))


def predict_rf_hallucination_index(vector_scores_answer, bm25_scores_answer, sources, nli_labels) -> float | None:
    artifact = _rf_artifact()
    if artifact is None:
        return None
    a_v_mean, a_v_max, a_v_min, a_v_std = _safe_stats(vector_scores_answer)
    a_b_mean, a_b_max, a_b_min, a_b_std = _safe_stats(bm25_scores_answer)
    q_v = [float(s["vector_score"]) for s in sources if s.get("vector_score") is not None]
    q_b = [float(s["bm25_score"]) for s in sources if s.get("bm25_score") is not None]
    q_r = [float(s["rerank_score"]) for s in sources if s.get("rerank_score") is not None]
    q_v_mean, q_v_max, q_v_min, q_v_std = _safe_stats(q_v)
    q_b_mean, q_b_max, q_b_min, q_b_std = _safe_stats(q_b)
    q_r_mean, q_r_max, q_r_min, q_r_std = _safe_stats(q_r)
    contradiction = sum(1 for x in nli_labels if str(x).lower() == "contradiction")
    entailment = sum(1 for x in nli_labels if str(x).lower() == "entailment")
    features = [
        a_v_mean, a_b_mean, q_v_mean, q_b_mean, q_r_mean,
        a_v_max, a_b_max, a_v_min, a_b_min, a_v_std, a_b_std,
        q_v_max, q_b_max, q_r_max, q_v_min, q_b_min, q_r_min,
        q_v_std, q_b_std, q_r_std, float(contradiction), float(entailment),
    ]
    try:
        proba = artifact["model"].predict_proba(np.array([features], dtype=float))[:, 1]
        return round(float(proba[0]) * 100, 0) if len(proba) else None
    except Exception:  # noqa: BLE001
        return None


# =========================================================================== #
# CONFIG  (mirror of config.py)
# =========================================================================== #
_DEFAULT_CUTOFFS = EmbeddingCutoffs(best_q=0.10, worst_q=0.80, best_ac=0.10, worst_ac=0.80)


@dataclass
class DatasetConfig:
    dataset_name: str
    embedding_model: str
    reranker: str
    use_bm25: bool
    use_reranker: bool
    documents_language: str
    hi_by_equation: bool
    scoring: ScoringConfig
    cutoffs: EmbeddingCutoffs
    reranker_cutoff: float


# Per-corpus deviations from the packaged defaults (mirror of config.py / the
# dataset_config CSV) so they apply even without MYGPT_WAREHOUSE_ID.
_PACKAGED_OVERRIDES = {
    "health_policies_fr": {"use_reranker": False, "documents_language": "french"},
    "health_policies_es": {"use_reranker": False, "documents_language": "spanish"},
}


def _config_fallback(dataset_name: str) -> DatasetConfig:
    ov = _PACKAGED_OVERRIDES.get(dataset_name, {})
    return DatasetConfig(
        dataset_name=dataset_name, embedding_model="bge-m3",
        reranker="cross-encoder/qnli-electra-base", use_bm25=True,
        use_reranker=ov.get("use_reranker", True),
        documents_language=ov.get("documents_language", "english"), hi_by_equation=True,
        scoring=ScoringConfig(), cutoffs=_DEFAULT_CUTOFFS, reranker_cutoff=0.10,
    )


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "t", "yes")


def load_dataset_config(dataset_name: str, workspace) -> DatasetConfig:
    """Read tunable config from the Delta tables if a warehouse is configured; else
    fall back to packaged MyGPT defaults so the endpoint always serves."""
    warehouse_id = os.environ.get("MYGPT_WAREHOUSE_ID")
    if not warehouse_id:
        return _config_fallback(dataset_name)
    try:
        def q(sql: str) -> list[dict]:
            resp = workspace.statement_execution.execute_statement(
                warehouse_id=warehouse_id, statement=sql, wait_timeout="30s"
            )
            cols = [c.name for c in resp.manifest.schema.columns]
            return [dict(zip(cols, r)) for r in (resp.result.data_array or [])]

        ds = q(f"SELECT * FROM {CATALOG_SCHEMA}.dataset_config WHERE dataset_name = '{dataset_name}'")
        if not ds:
            return _config_fallback(dataset_name)
        d = ds[0]
        cut = q(f"SELECT * FROM {CATALOG_SCHEMA}.embedding_model_cutoffs WHERE model_name = '{d['embedding_model']}'")
        c = cut[0] if cut else None
        rk = q(f"SELECT cutoff_score FROM {CATALOG_SCHEMA}.reranker_config WHERE model_name = '{d['reranker']}'")
        return DatasetConfig(
            dataset_name=dataset_name, embedding_model=str(d["embedding_model"]), reranker=str(d["reranker"]),
            use_bm25=_as_bool(d["use_bm25"]), use_reranker=_as_bool(d["use_reranker"]),
            documents_language=str(d["documents_language"]), hi_by_equation=_as_bool(d["HI_by_equation"]),
            scoring=ScoringConfig(
                a_qsem=float(d["coefficient_a_Qsem"]), b_qkey=float(d["coefficient_b_Qkey"]), c_qrank=float(d["coefficient_c_Qrank"]),
                x_asem=float(d["coefficient_x_Asem"]), y_akey=float(d["coefficient_y_Akey"]), z_arank=float(d["coefficient_z_Arank"]),
                p_qrs=float(d["coefficient_p_QRS"]), q_ars=float(d["coefficient_q_ARS"]),
            ),
            cutoffs=EmbeddingCutoffs(
                best_q=float(c["best_distance_q"]) if c else _DEFAULT_CUTOFFS.best_q,
                worst_q=float(c["worst_distance_q"]) if c else _DEFAULT_CUTOFFS.worst_q,
                best_ac=float(c["best_distance_ac"]) if c else _DEFAULT_CUTOFFS.best_ac,
                worst_ac=float(c["worst_distance_ac"]) if c else _DEFAULT_CUTOFFS.worst_ac,
            ),
            reranker_cutoff=float(rk[0]["cutoff_score"]) if rk else 0.10,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[config] fallback to defaults: {type(exc).__name__}: {exc}")
        return _config_fallback(dataset_name)


# =========================================================================== #
# AGENT
# =========================================================================== #
class MyGPTRagAgent(ResponsesAgent):
    def __init__(self) -> None:
        self._cfg_cache: dict[str, DatasetConfig] = {}
        self._reranker = None
        self._nli = None
        self._ws = None

    def _workspace(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient

            self._ws = WorkspaceClient()
        return self._ws

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

    def _config(self, dataset: str) -> DatasetConfig:
        if dataset not in self._cfg_cache:
            self._cfg_cache[dataset] = load_dataset_config(dataset, self._workspace())
        return self._cfg_cache[dataset]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = self._workspace().api_client.do(
            "POST", f"/serving-endpoints/{EMBED_ENDPOINT}/invocations", body={"input": texts}
        )
        return _parse_embeddings(resp, n=len(texts))

    def _retrieve_traced(self, query: str, corpus: str | None, focus: dict, k: int) -> list[dict]:
        """Run retrieval inside a RETRIEVER span whose outputs use MLflow's Document
        shape (page_content + metadata.doc_uri), so built-in judges
        (RetrievalGroundedness / RetrievalRelevance) can read the retrieved context off
        the trace. Returns the raw source dicts the scoring pipeline works on."""
        with mlflow.start_span(name="retrieve", span_type="RETRIEVER") as span:
            span.set_inputs({"query": query, "corpus": corpus, "focus": focus, "k": k})
            sources = self._retrieve(query, corpus, focus, k)
            span.set_outputs([
                {
                    "page_content": s["context"],
                    "metadata": {
                        "doc_uri": s["source_name"], "chunk_id": s["chunk_id"], "corpus": s["corpus"],
                        "section": s["section"], "page": s["page"], "similarity": s["similarity"],
                    },
                }
                for s in sources
            ])
        return sources

    def _retrieve(self, query: str, corpus: str | None, focus: dict, k: int) -> list[dict]:
        filters: dict[str, Any] = {}
        if corpus:
            filters["corpus"] = corpus
        docs = focus.get("documents") or []
        section = focus.get("section")
        if docs:
            filters["source_name"] = docs
        if section and section != "all":
            filters["section"] = section
        payload: dict[str, Any] = {
            "columns": ["chunk_id", "source_name", "corpus", "section", "page", "chunk_text"],
            "query_text": query,
            "num_results": k,
            "query_type": "HYBRID",
        }
        if filters:
            payload["filters_json"] = json.dumps(filters)
        try:
            # Query via the serving endpoint's injected credentials (same REST path
            # used for embeddings/LLM) — more reliable in Model Serving than the
            # VectorSearchClient credential discovery. Also lets the MLflow log-time
            # validation predict degrade cleanly when VS is unreachable.
            res = self._workspace().api_client.do(
                "POST", f"/api/2.0/vector-search/indexes/{VS_INDEX}/query", body=payload
            )
        except Exception as exc:  # noqa: BLE001
            self._last_retrieve_error = f"{type(exc).__name__}: {exc}"
            print(f"[retrieve] vector search unavailable: {self._last_retrieve_error}")
            return []
        self._last_retrieve_error = None
        cols = [c["name"] for c in (res.get("manifest", {}) or {}).get("columns", [])]
        rows = res.get("result", {}).get("data_array", []) or []
        sources: list[dict] = []
        for rank, row in enumerate(rows, start=1):
            r = dict(zip(cols, row))
            sim = float(r.get("score", 0.0))
            sources.append({
                "chunk_id": r.get("chunk_id"), "source_name": r.get("source_name") or "",
                "corpus": r.get("corpus") or "", "section": r.get("section") or "", "page": None if r.get("page") is None else int(r["page"]),
                "context": str(r.get("chunk_text") or ""), "similarity": sim,
                "vector_distance_raw": round(1.0 - sim, 4), "bm25_score": round(sim, 4), "rank": rank,
            })
        return sources

    @mlflow.trace(span_type="RERANKER")
    def _rerank(self, query: str, sources: list[dict], cfg: DatasetConfig) -> list[dict]:
        if not sources or not cfg.use_reranker:
            for s in sources:
                s["rerank_score"] = 0.0
            return sources
        scores = self._reranker_model().predict([[query, s["context"]] for s in sources])
        kept = []
        for s, sc in zip(sources, scores):
            s["rerank_score"] = round(float(sc), 3)
            if s["rerank_score"] > cfg.reranker_cutoff:
                kept.append(s)
        kept.sort(key=lambda s: s["rerank_score"], reverse=True)
        for i, s in enumerate(kept, start=1):
            s["rank"] = i
        return kept or sources

    @mlflow.trace(span_type="PARSER")
    def _score_question(self, sources: list[dict], cfg: DatasetConfig) -> tuple[float, float, list[dict]]:
        distances = [s["vector_distance_raw"] for s in sources]
        semantic, normalized = question_semantic_score(distances, cfg.cutoffs)
        for s, nd in zip(sources, normalized):
            s["vector_score"] = round(nd, 3)
        keyword = (sum(s["bm25_score"] for s in sources) / len(sources)) if (sources and cfg.use_bm25) else 0.0
        rerank = (sum(s["rerank_score"] for s in sources) / len(sources)) if sources else 0.0
        qrs = question_relevance_score(semantic, keyword, rerank, cfg.scoring)
        return qrs, semantic, sources

    @mlflow.trace(span_type="LLM")
    def _generate(self, query: str, sources: list[dict], model_alias: str, instructions: str = "") -> str:
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
        if instructions:
            # Per-request answer-format instructions (e.g. the MyGPT-evaluations kinase
            # JS-array formats), appended so the grounding rules above still apply.
            prompt += f"\n\nAnswer format instructions: {instructions}"
        endpoint = MODEL_ALIASES.get(model_alias, MODEL_ALIASES["sonnet"])
        # Chat via the serving endpoint's injected credentials (REST), same as
        # retrieval/embeddings — avoids the databricks-sdk[openai] dependency.
        resp = self._workspace().api_client.do(
            "POST", f"/serving-endpoints/{endpoint}/invocations",
            body={"messages": [{"role": "user", "content": prompt}]},
        )
        choices = resp.get("choices") or []
        if not choices:
            return ""
        content = (choices[0].get("message") or {}).get("content") or ""
        if isinstance(content, list):
            # Reasoning-capable models (e.g. Claude Sonnet 5) can return content blocks —
            # [{"type": "reasoning", "summary": [...]}, {"type": "text", "text": ...}].
            # Keep only the answer text; passing the blocks on broke _embed (bge-m3
            # rejects dict inputs) and leaked reasoning into the answer.
            content = "".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") in ("text", "output_text")
            )
        return str(content).strip()

    @mlflow.trace(span_type="PARSER")
    def _score_answer(self, answer: str, sources: list[dict], question_semantic: float, cfg: DatasetConfig) -> dict:
        if not answer or not sources:
            return {"ars": 0.0, "hi_eq": 100.0, "hi_rf": None, "nli": []}
        vectors = self._embed([answer] + [s["context"] for s in sources])
        a_vec, s_vecs = vectors[0], vectors[1:]
        answer_distances = [round(1.0 - _cosine(a_vec, sv), 4) for sv in s_vecs]
        answer_semantic, answer_vec_scores = answer_vector_scores(answer_distances, cfg.cutoffs)

        nli_labels: list[str] = []
        if cfg.documents_language == "english":
            label_map = ["contradiction", "entailment", "neutral"]
            preds = self._nli_model().predict([(answer, s["context"]) for s in sources])
            nli_labels = [label_map[int(p.argmax())] for p in preds]

        rerank_mean_100 = (sum(s["rerank_score"] for s in sources) / len(sources)) * 100
        answer_keyword_100 = (sum(s["bm25_score"] for s in sources) / len(sources)) * 100 if cfg.use_bm25 else 0.0
        ars = answer_relevance_score(answer_semantic, answer_keyword_100, rerank_mean_100, cfg.scoring)
        hi_eq = hallucination_index_equation(question_semantic, ars, cfg.scoring)
        hi_rf = predict_rf_hallucination_index(
            answer_vec_scores, [s["bm25_score"] for s in sources], sources, nli_labels
        )
        return {"ars": ars, "hi_eq": hi_eq, "hi_rf": hi_rf, "nli": nli_labels}

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        ci = request.custom_inputs or {}
        # corpus (optional) isolates retrieval to one top-level volume folder; absent =
        # search every corpus (the app's default). Config defaults to the corpus' row.
        corpus = ci.get("corpus") or None
        dataset = ci.get("dataset") or corpus or "mygpt"
        model_alias = ci.get("model", "sonnet")
        focus = ci.get("focus", {}) or {}
        k = int(ci.get("k", 5))
        instructions = str(ci.get("instructions") or "")
        query = _last_user_text(request)
        cfg = self._config(dataset)
        model_endpoint = MODEL_ALIASES.get(model_alias, MODEL_ALIASES["sonnet"])
        try:
            # Tags let eval runs and production monitoring slice traces per corpus/model.
            mlflow.update_current_trace(tags={
                "mygpt.corpus": corpus or "all", "mygpt.dataset": dataset, "mygpt.model": model_endpoint,
            })
        except Exception:  # noqa: BLE001  (no active trace, e.g. tracing disabled)
            pass

        sources = self._retrieve_traced(query, corpus, focus, k)
        if not sources:
            item = self.create_text_output_item(
                "I couldn't find anything relevant in the indexed documents.", str(uuid.uuid4())
            )
            return ResponsesAgentResponse(output=[item], custom_outputs={
                "model": model_endpoint, "corpus": corpus,
                "qrs": 0, "ars": 0, "hi_eq": 100, "hi_rf": None, "sources": [],
                # Surface the caught retrieval error instead of silently returning [].
                "retrieve_error": getattr(self, "_last_retrieve_error", None),
            })

        sources = self._rerank(query, sources, cfg)
        qrs, question_semantic, sources = self._score_question(sources, cfg)
        answer = self._generate(query, sources, model_alias, instructions)
        ans = self._score_answer(answer, sources, question_semantic, cfg)
        for s, label in zip(sources, ans["nli"]):
            s["nli"] = label

        item = self.create_text_output_item(answer, str(uuid.uuid4()))
        return ResponsesAgentResponse(
            output=[item],
            custom_outputs={
                "model": model_endpoint, "corpus": corpus,
                "qrs": qrs, "ars": ans["ars"], "hi_eq": ans["hi_eq"], "hi_rf": ans["hi_rf"],
                "sources": [
                    {
                        "source_name": s["source_name"], "corpus": s["corpus"], "section": s["section"], "page": s["page"],
                        "snippet": s["context"][:240], "vector_score": s.get("vector_score"),
                        "bm25_score": s.get("bm25_score"), "rerank_score": s.get("rerank_score"), "nli": s.get("nli"),
                    }
                    for s in sources
                ],
            },
        )


def _parse_embeddings(resp: Any, n: int) -> list[list[float]]:
    if isinstance(resp, dict):
        if "data" in resp:
            return [d["embedding"] for d in resp["data"]]
        for key in ("predictions", "embeddings", "outputs"):
            if key in resp:
                resp = resp[key]
                break
    if isinstance(resp, list):
        if resp and isinstance(resp[0], (int, float)) and n == 1:
            return [list(resp)]
        return [list(v) for v in resp]
    raise ValueError(f"Unrecognized embeddings response: {type(resp)}")


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
            if isinstance(content, list):
                return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
            return str(content)
    return ""


mlflow.models.set_model(MyGPTRagAgent())
