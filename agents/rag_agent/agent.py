"""MyGPT retrieval + scoring + generation as a self-contained MLflow ResponsesAgent.

Deployed to its own Model Serving endpoint (Mosaic AI Agent Framework). Owns the
full MyGPT-grade pipeline and returns the answer plus QRS / ARS / Hallucination-Index
trust metrics and per-source scores via ``custom_outputs``:

  1. retrieve   AI Search (ANN + hybrid) with document/section focus filters — or, for
                Phase-4 upload libraries, the chunks the app already retrieved from
                Lakebase under RLS (``custom_inputs.sources``; see sources_from_inputs)
  2. rerank     cross-encoder/qnli-electra-base for English, mmarco-mMiniLMv2 for other
                languages (query <-> source relevance)
  3. QRS        semantic (per-model distance calibration) + BM25 + rerank
  4. generate   FMAPI chat (Claude / Llama / GPT-OSS)
  5. NLI        cross-encoder/nli-deberta-v3-base (answer <-> source entailment; English only)
  6. ARS + HI   answer relevance (stemmed answer-vs-context BM25) + hallucination index
                (equation; RF toggle)

Phase 6 (multilingual): when the dataset's documents aren't in the question's language,
the question is translated into the document language (``mygpt-translate``) for retrieval
and rerank, and the answer is translated the same way for ARS scoring only — MyGPT's
translategemma flow. The user-facing answer stays in the question's language.

The scoring math, config loader, and random-forest HI are INLINED here (rather than
imported from scoring.py / config.py / hi_random_forest.py) because MLflow imports
this file in isolation when validating a code-based model, so sibling imports are
not resolvable at log time. ``scoring.py`` + ``test_scoring.py`` remain the verified
reference these inlined functions mirror — keep them in sync.

Fidelity = Balanced (see PLAN.md): the question-side BM25 term is derived from AI Search
hybrid scores rescored per source rather than a separate full-corpus bm25s index. The
answer-side term is a real stemmed BM25 of the answer over the retrieved contexts, as in
MyGPT (``answer_bm25``).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
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
# Phase 6: query/answer translation goes through its own Unity AI Gateway endpoint
# (mygpt-translate -> databricks-gemma-3-12b, the base model of MyGPT's translategemma).
TRANSLATE_ENDPOINT = os.environ.get("MYGPT_TRANSLATE_ENDPOINT", "mygpt-translate")
RERANK_MODEL = "cross-encoder/qnli-electra-base"  # English (MyGPT default)
RERANK_MODEL_MULTILINGUAL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # MyGPT's non-English default
RERANK_CUTOFFS = {RERANK_MODEL: 0.10, RERANK_MODEL_MULTILINGUAL: 0.05}
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# MyGPT's documents_language choices -> (full name, ISO code) for the translator prompt.
# "unknown" has no entry: such libraries get the multilingual reranker but no translation.
LANGUAGES = {
    "english": ("English", "en"), "french": ("French", "fr"), "spanish": ("Spanish", "es"),
    "portuguese": ("Portuguese", "pt"), "german": ("German", "de"), "italian": ("Italian", "it"),
    "dutch": ("Dutch", "nl"), "arabic": ("Arabic", "ar"), "hindi": ("Hindi", "hi"),
    "hungarian": ("Hungarian", "hu"),
}
# Governed corpora whose documents aren't English (mirror of dataset_config). Used for the
# packaged config and to pick a per-source reranker in the mixed "all corpora" scope.
CORPUS_LANGUAGE = {
    "health_policies_fr": "french", "health_policies_es": "spanish", "health_policies_hi": "hindi",
}


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


# Words of 2+ characters, as in bm25s' default token pattern, but Devanagari vowel signs
# and viramas (Unicode Mn/Mc, which \w doesn't match) stay inside the word.
_TOKEN_RE = re.compile(r"[\wऀ-ॿ]{2,}")
_STEMMERS: dict[str, Any] = {}


def _stemmer(language: str):
    """PyStemmer (Snowball) stemmer for a MyGPT language name; None if unsupported."""
    if language not in _STEMMERS:
        try:
            import Stemmer

            _STEMMERS[language] = Stemmer.Stemmer(language)
        except Exception:  # noqa: BLE001  ("unknown", or no PyStemmer)
            _STEMMERS[language] = None
    return _STEMMERS[language]


def bm25_tokenize(text: str, language: str) -> list[str]:
    tokens = _TOKEN_RE.findall(str(text).lower())
    stemmer = _stemmer(language)
    return stemmer.stemWords(tokens) if stemmer else tokens


def answer_bm25_scores(answer: str, contexts: Sequence[str], language: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Raw BM25 of the answer against each context (same order), with the contexts as the
    corpus — MyGPT's get_answer_distance_by_context_bm25 (bm25s, method "lucene",
    PyStemmer for the documents' language), computed in memory."""
    docs = [bm25_tokenize(c, language) for c in contexts]
    n = len(docs)
    if n == 0:
        return []
    avgdl = (sum(len(d) for d in docs) / n) or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    query = set(bm25_tokenize(answer, language))
    scores = []
    for d in docs:
        tf: dict[str, int] = {}
        for t in d:
            if t in query:
                tf[t] = tf.get(t, 0) + 1
        norm = k1 * (1 - b + b * len(d) / avgdl)
        scores.append(sum(
            math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) * f / (f + norm) for t, f in tf.items()
        ))
    return scores


def answer_keyword_score(raw_scores: Sequence[float], cfg: ScoringConfig) -> tuple[float, list[float]]:
    """Per-source [0, 1] BM25 scores (min-max on bm25_best/worst) and their mean x100."""
    if not raw_scores:
        return 0.0, []
    scores = [round(v, 3) for v in min_max_normalization([round(r, 3) for r in raw_scores], cfg.bm25_best, cfg.bm25_worst)]
    return round(sum(scores) / len(scores) * 100, 2), scores


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
    translate_query: bool = False
    answer_bm25: bool = True


def dataset_language(dataset_name: str) -> str:
    """Documents language of a packaged dataset: a governed corpus (CORPUS_LANGUAGE) or an
    upload-library config row ``default_<language>``; everything else is English."""
    if dataset_name.startswith("default_"):
        return dataset_name[len("default_"):]
    return CORPUS_LANGUAGE.get(dataset_name, "english")


def reranker_for(language: str) -> str:
    return RERANK_MODEL if language == "english" else RERANK_MODEL_MULTILINGUAL


def _config_fallback(dataset_name: str) -> DatasetConfig:
    """Packaged defaults (mirror of config.py / the dataset_config CSV), derived from the
    dataset's language so they apply even without MYGPT_WAREHOUSE_ID."""
    language = dataset_language(dataset_name)
    reranker = reranker_for(language)
    return DatasetConfig(
        dataset_name=dataset_name, embedding_model="bge-m3", reranker=reranker, use_bm25=True,
        use_reranker=True, documents_language=language, hi_by_equation=True,
        scoring=ScoringConfig(), cutoffs=_DEFAULT_CUTOFFS, reranker_cutoff=RERANK_CUTOFFS[reranker],
        translate_query=language in LANGUAGES and language != "english", answer_bm25=True,
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
            reranker_cutoff=float(rk[0]["cutoff_score"]) if rk else RERANK_CUTOFFS.get(str(d["reranker"]), 0.10),
            translate_query=_as_bool(d.get("translate_query", False)),
            answer_bm25=_as_bool(d.get("answer_bm25", True)),
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
        self._rerankers: dict[str, Any] = {}
        self._nli = None
        self._ws = None

    def _workspace(self):
        if self._ws is None:
            from databricks.sdk import WorkspaceClient

            self._ws = WorkspaceClient()
        return self._ws

    def _reranker_model(self, name: str = RERANK_MODEL):
        if name not in self._rerankers:
            import torch
            from sentence_transformers import CrossEncoder

            self._rerankers[name] = CrossEncoder(name, activation_fn=torch.nn.Sigmoid())
        return self._rerankers[name]

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

    def _retrieve_traced(
        self, query: str, corpus: str | None, focus: dict, k: int, provided: list[dict] | None = None
    ) -> list[dict]:
        """Run retrieval inside a RETRIEVER span whose outputs use MLflow's Document
        shape (page_content + metadata.doc_uri), so built-in judges
        (RetrievalGroundedness / RetrievalRelevance) can read the retrieved context off
        the trace. Returns the raw source dicts the scoring pipeline works on.
        ``provided`` (upload libraries) skips AI Search but keeps the same span."""
        with mlflow.start_span(name="retrieve", span_type="RETRIEVER") as span:
            span.set_inputs({"query": query, "corpus": corpus, "focus": focus, "k": k,
                             "provided": provided is not None})
            sources = provided if provided is not None else self._retrieve(query, corpus, focus, k)
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

    def _translate(self, text: str, source_language: str, target_language: str) -> tuple[str, str | None]:
        """Translate with MyGPT's translategemma system prompt. Returns (text, error); any
        failure falls back to the original text so translation never fails a request."""
        with mlflow.start_span(name="translate", span_type="LLM") as span:
            span.set_inputs({"text": text, "source": source_language, "target": target_language})
            src_full, src_code = LANGUAGES.get(source_language, (source_language.title(), source_language[:2]))
            tgt_full, tgt_code = LANGUAGES[target_language]
            system = (
                f"You are a professional {src_full} ({src_code}) to {tgt_full} ({tgt_code}) translator. "
                f"Your goal is to accurately convey the meaning and nuances of the original {src_full} text "
                f"while adhering to {tgt_full} grammar, vocabulary, and cultural sensitivities. Produce only "
                f"the {tgt_full} translation, without any additional explanations or commentary. If the text "
                f"is already in {tgt_full}, return it unchanged. Please translate the following {src_full} "
                f"text into {tgt_full}: "
            )
            try:
                resp = self._workspace().api_client.do(
                    "POST", f"/serving-endpoints/{TRANSLATE_ENDPOINT}/invocations",
                    body={"messages": [{"role": "system", "content": system}, {"role": "user", "content": text}],
                          "temperature": 0},
                )
                out = _message_text(resp)
                if not out:
                    raise ValueError("empty translation")
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                print(f"[translate] falling back to the original text: {error}")
                span.set_outputs({"text": text, "error": error})
                return text, error
            span.set_outputs({"text": out})
            return out, None

    @mlflow.trace(span_type="RERANKER")
    def _rerank(self, query: str, sources: list[dict], cfg: DatasetConfig, per_source_language: bool = False) -> list[dict]:
        """Rerank with the dataset's reranker. In the mixed "all corpora" scope
        (``per_source_language``) each source uses the reranker for its corpus' language."""
        if not sources or not cfg.use_reranker:
            for s in sources:
                s["rerank_score"] = 0.0
            return sources
        groups: dict[str, list[dict]] = {}
        for s in sources:
            name = reranker_for(CORPUS_LANGUAGE.get(s["corpus"], "english")) if per_source_language else cfg.reranker
            groups.setdefault(name, []).append(s)
        kept = []
        for name, group in groups.items():
            cutoff = cfg.reranker_cutoff if name == cfg.reranker else RERANK_CUTOFFS.get(name, 0.10)
            scores = self._reranker_model(name).predict([[query, s["context"]] for s in group])
            for s, sc in zip(group, scores):
                s["rerank_score"] = round(float(sc), 3)
                s["reranker"] = name
                if s["rerank_score"] > cutoff:
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
            "do not contain the answer, say you couldn't find it in the documents. Answer in the "
            "language of the question, even when the sources are in another language.\n\n"
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
        return _message_text(resp)

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
        if cfg.use_bm25 and cfg.answer_bm25:
            # Phase 6: stemmed BM25 of the answer over the retrieved contexts (MyGPT).
            raw = answer_bm25_scores(answer, [s["context"] for s in sources], cfg.documents_language)
            answer_keyword_100, answer_bm25 = answer_keyword_score(raw, cfg.scoring)
            for s, v in zip(sources, answer_bm25):
                s["answer_bm25_score"] = v
        else:
            # Pre-Phase-6 behaviour: reuse the question-side hybrid scores.
            answer_bm25 = [s["bm25_score"] for s in sources]
            answer_keyword_100 = (sum(answer_bm25) / len(answer_bm25)) * 100 if cfg.use_bm25 else 0.0
        ars = answer_relevance_score(answer_semantic, answer_keyword_100, rerank_mean_100, cfg.scoring)
        hi_eq = hallucination_index_equation(question_semantic, ars, cfg.scoring)
        hi_rf = predict_rf_hallucination_index(answer_vec_scores, answer_bm25, sources, nli_labels)
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
        include_context = bool(ci.get("include_context"))
        query = _last_user_text(request)
        # Phase 4: upload libraries arrive pre-retrieved (the app searched Lakebase under RLS).
        provided = sources_from_inputs(ci["sources"], k) if ci.get("sources") is not None else None
        self._last_retrieve_error = None
        cfg = self._config(dataset)
        model_endpoint = MODEL_ALIASES.get(model_alias, MODEL_ALIASES["sonnet"])
        # Phase 6: questions are assumed English (MyGPT's assumption) unless the caller says
        # otherwise; `translate: false` turns translation off for eval A/B runs.
        query_language = str(ci.get("query_language") or "english").lower()
        doc_language = cfg.documents_language
        translate = (
            cfg.translate_query and _as_bool(ci.get("translate", True))
            and doc_language in LANGUAGES and doc_language != query_language
        )
        # The mixed "all corpora" scope has no single document language: no translation,
        # and each source is reranked for its own corpus' language.
        mixed_scope = corpus is None and provided is None
        try:
            # Tags let eval runs and production monitoring slice traces per corpus/model.
            mlflow.update_current_trace(tags={
                "mygpt.corpus": corpus or "all", "mygpt.dataset": dataset, "mygpt.model": model_endpoint,
                "mygpt.retrieval": "provided" if provided is not None else "ai_search",
                "mygpt.language": doc_language, "mygpt.translated": str(translate).lower(),
            })
        except Exception:  # noqa: BLE001  (no active trace, e.g. tracing disabled)
            pass

        translated_query, translate_errors = None, []
        search_query = query
        if translate:
            translated_query, err = self._translate(query, query_language, doc_language)
            search_query = translated_query
            if err:
                translate_errors.append(f"query: {err}")
        language_outputs = {
            "documents_language": doc_language, "translated_query": translated_query,
            "translate_error": "; ".join(translate_errors) or None,
        }

        sources = self._retrieve_traced(search_query, corpus, focus, k, provided)
        trace_id = _active_trace_id()
        if not sources:
            item = self.create_text_output_item(
                "I couldn't find anything relevant in the indexed documents.", str(uuid.uuid4())
            )
            return ResponsesAgentResponse(output=[item], custom_outputs={
                "model": model_endpoint, "corpus": corpus,
                "qrs": 0, "ars": 0, "hi_eq": 100, "hi_rf": None, "sources": [], "trace_id": trace_id,
                # Surface the caught retrieval error instead of silently returning [].
                "retrieve_error": getattr(self, "_last_retrieve_error", None),
                **language_outputs, "translated_answer": None,
            })

        sources = self._rerank(search_query, sources, cfg, per_source_language=mixed_scope)
        qrs, question_semantic, sources = self._score_question(sources, cfg)
        # The model sees the original question, so it answers in the user's language.
        answer = self._generate(query, sources, model_alias, instructions)
        # ...and the answer is scored in the documents' language (MyGPT scores the
        # translated answer: embedding distance, BM25 and NLI).
        translated_answer, scored_answer = None, answer
        if translate and answer:
            translated_answer, err = self._translate(answer, query_language, doc_language)
            scored_answer = translated_answer
            if err:
                translate_errors.append(f"answer: {err}")
                language_outputs["translate_error"] = "; ".join(translate_errors)
        ans = self._score_answer(scored_answer, sources, question_semantic, cfg)
        for s, label in zip(sources, ans["nli"]):
            s["nli"] = label

        item = self.create_text_output_item(answer, str(uuid.uuid4()))
        return ResponsesAgentResponse(
            output=[item],
            custom_outputs={
                "model": model_endpoint, "corpus": corpus,
                "qrs": qrs, "ars": ans["ars"], "hi_eq": ans["hi_eq"], "hi_rf": ans["hi_rf"],
                # lets the app attach user feedback to this trace (Phase 4)
                "trace_id": trace_id,
                **language_outputs, "translated_answer": translated_answer,
                "reranker": cfg.reranker if cfg.use_reranker else None,
                "sources": [
                    {
                        "source_name": s["source_name"], "corpus": s["corpus"], "section": s["section"], "page": s["page"],
                        "snippet": s["context"][:240], "vector_score": s.get("vector_score"),
                        "bm25_score": s.get("bm25_score"), "rerank_score": s.get("rerank_score"), "nli": s.get("nli"),
                        "answer_bm25_score": s.get("answer_bm25_score"), "reranker": s.get("reranker"),
                        # Phase 5: the Supervisor asks for the full chunk text so its own
                        # RETRIEVER span can feed the groundedness judges.
                        **({"chunk_id": s["chunk_id"], "context": s["context"]} if include_context else {}),
                    }
                    for s in sources
                ],
            },
        )


MAX_PROVIDED_SOURCES = 10
MAX_PROVIDED_CHARS = 4000


def _clamp(value: Any, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    return lo if v != v else min(hi, max(lo, v))  # NaN -> lo


def sources_from_inputs(raw: Any, k: int) -> list[dict]:
    """Pre-retrieved chunks (``custom_inputs.sources``) -> the exact source-dict shape
    ``_retrieve`` builds from AI Search, so rerank / QRS / generate / ARS / HI run unchanged.

    The app sends cosine ``similarity`` (pgvector) and a [0, 1] keyword ``bm25_score``
    (normalized ts_rank_cd). Malformed entries are dropped; at most k (<= 10) are kept and
    chunk text is truncated, since this input crosses the serving-endpoint boundary.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("chunk_text") or item.get("context") or "").strip()
        if not text:
            continue
        sim = round(_clamp(item.get("similarity"), -1.0, 1.0), 4)
        page = item.get("page")
        out.append({
            "chunk_id": str(item.get("chunk_id") or ""),
            "source_name": str(item.get("source_name") or ""),
            "corpus": str(item.get("corpus") or ""),
            "section": str(item.get("section") or ""),
            "page": int(page) if isinstance(page, (int, float)) and not isinstance(page, bool) else None,
            "context": text[:MAX_PROVIDED_CHARS],
            "similarity": sim,
            "vector_distance_raw": round(1.0 - sim, 4),
            "bm25_score": round(_clamp(item.get("bm25_score"), 0.0, 1.0), 4),
            "rank": len(out) + 1,
        })
        if len(out) >= max(1, min(int(k), MAX_PROVIDED_SOURCES)):
            break
    return out


def _message_text(resp: Any) -> str:
    """Answer text of a chat-completions response."""
    choices = (resp or {}).get("choices") or []
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


def _active_trace_id() -> str | None:
    try:
        return mlflow.get_active_trace_id()
    except Exception:  # noqa: BLE001  (tracing disabled / older MLflow)
        return None


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
