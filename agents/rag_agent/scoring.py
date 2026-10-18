"""Faithful port of St. Jude MyGPT's retrieval/answer trust scoring.

This module reproduces the exact QRS / ARS / Hallucination-Index math from the
reference implementation (``reference/MyGPT/backend/testdb/views``) so the
Databricks ResponsesAgent returns MyGPT-grade trust metrics. It is deliberately
dependency-light (numpy only) and side-effect free: the ML pieces (cross-encoder
rerank, NLI entailment, random-forest HI) are computed by the agent and passed in
as plain numbers/labels, which keeps this module unit-testable off-cluster.

Source map (verbatim behaviour preserved):
  - ``helpers.min_max_normalization``            -> ``min_max_normalization``
  - ``analytics.get_relevance_score``            -> ``question_semantic_score``
  - ``apis.get_context`` QRS assembly            -> ``question_relevance_score``
  - ``apis.save_answer`` ARS vector scoring       -> ``answer_vector_scores``
  - ``apis.save_answer`` ARS assembly             -> ``answer_relevance_score``
  - ``apis.save_answer`` HI-by-equation           -> ``hallucination_index_equation``
  - ``bm25_utils.get_answer_distance_by_context_bm25`` + the ``apis.save_answer``
    keyword term                                  -> ``answer_bm25_scores`` / ``answer_keyword_score``

Known MyGPT quirks preserved on purpose (see inline notes):
  - QRS uses the mean *normalized* BM25 score (0-1) for the keyword term, whereas
    ARS uses the mean BM25 score expressed 0-100 then divided by 100. The net
    keyword magnitude differs between the two formulas; we keep both as-is.
  - The hallucination-index equation is driven by the question *semantic* score
    (``question.relevance_score`` in MyGPT), not the fully-normalized QRS. We
    default to that faithful behaviour; callers may override.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# Configuration (ports MyGPT's per-dataset Dataset + EmbeddingModel columns).
# Defaults are MyGPT's defaults; per-dataset overrides come from the Delta
# config tables (dataset_config / embedding_model_cutoffs).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScoringConfig:
    # QRS coefficients (Dataset.coefficient_{a_Qsem,b_Qkey,c_Qrank}).
    a_qsem: float = 4.0
    b_qkey: float = -4.0
    c_qrank: float = -1.0
    qrs_raw_max: float = 3.03   # relevance_score_max
    qrs_raw_min: float = -5.5   # relevance_score_min

    # ARS coefficients (Dataset.coefficient_{x_Asem,y_Akey,z_Arank}).
    x_asem: float = 5.0
    y_akey: float = -2.0
    z_arank: float = 0.0
    ars_raw_max: float = 3.65
    ars_raw_min: float = -6.42

    # Hallucination-index weights (Dataset.coefficient_{p_QRS,q_ARS}).
    p_qrs: float = 1.0
    q_ars: float = 2.0
    hi_max: float = 0.8
    hi_min: float = 0.2

    # BM25 raw-score normalization window (apis.py best/worst_bm25_score).
    bm25_best: float = 20.0
    bm25_worst: float = 0.0


@dataclass(frozen=True)
class EmbeddingCutoffs:
    """Per-embedding-model distance calibration (EmbeddingModel.*_distance_*).

    ``best_*`` is the smaller (better) distance, ``worst_*`` the larger. These are
    recalibrated for whatever embedding model + similarity metric the index uses
    (MyGPT's literal Ollama/L2 numbers do not transfer to bge-m3). ``*_q`` is the
    question/retrieval window; ``*_ac`` is the answer-vs-context window.
    """
    best_q: float
    worst_q: float
    best_ac: float
    worst_ac: float


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #
def min_max_normalization(
    data: Sequence[float], best_val: float, worst_val: float, reverse: bool = False
) -> list[float]:
    """Port of ``helpers.min_max_normalization``.

    Assumes bigger is better; with ``reverse=True`` smaller is better (distances).
    No clamping — MyGPT lets values fall outside [0,1] and clamps later.
    """
    out: list[float] = []
    span = (worst_val - best_val) if reverse else (best_val - worst_val)
    for value in data:
        if reverse:
            out.append((worst_val - value) / span)
        else:
            out.append((value - worst_val) / span)
    return out


def _buffered_window(best: float, worst: float, lo_frac: float, hi_frac: float) -> tuple[float, float]:
    """Expand a [best, worst] distance window by fractional buffers.

    MyGPT's question path uses lo=0.1, hi=0.3 (asymmetric); the answer path uses
    lo=hi=0.05 (symmetric). Buffer size is a fraction of the raw (worst-best) span.
    """
    buffer = (worst - best) * lo_frac
    return best - buffer, worst + (worst - best) * hi_frac


# --------------------------------------------------------------------------- #
# Question side (QRS)
# --------------------------------------------------------------------------- #
def question_semantic_score(
    distances: Sequence[float], cutoffs: EmbeddingCutoffs
) -> tuple[float, list[float]]:
    """Port of ``analytics.get_relevance_score`` (question=True, use_default=True).

    Returns (semantic_score in 0-100, per-source normalized distances in ~0-1).
    """
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


def question_relevance_score(
    semantic_score: float,
    keyword_score: float,
    rerank_score: float,
    cfg: ScoringConfig = ScoringConfig(),
) -> float:
    """Port of the QRS assembly in ``apis.get_context``.

    - ``semantic_score``: 0-100 (from :func:`question_semantic_score`).
    - ``keyword_score``:  mean normalized BM25 across sources, 0-1.
    - ``rerank_score``:   mean cross-encoder reranked score across sources, 0-1.
    """
    raw = round(
        (semantic_score / 100 * cfg.a_qsem)
        + (keyword_score * cfg.b_qkey)
        + (rerank_score * cfg.c_qrank),
        2,
    )
    qrs = round(
        min_max_normalization([raw], cfg.qrs_raw_max, cfg.qrs_raw_min, reverse=False)[0] * 100,
        0,
    )
    return _clamp_0_100(qrs)


# --------------------------------------------------------------------------- #
# Answer side (ARS)
# --------------------------------------------------------------------------- #
def answer_vector_scores(
    answer_distances: Sequence[float], cutoffs: EmbeddingCutoffs
) -> tuple[float, list[float]]:
    """Port of the ARS vector-scoring block in ``apis.save_answer``.

    Uses the answer-vs-context window (``*_ac``) with a symmetric 5% buffer and the
    forward (bigger-is-better) normalization MyGPT applies to answer distances.
    Returns (answer semantic_score 0-100, per-source answer vector_scores).
    """
    best, worst = cutoffs.best_ac, cutoffs.worst_ac
    buffer = 0.05 * (worst - best)
    best, worst = best - buffer, worst + buffer
    if len(answer_distances) == 0:
        return 0.0, []
    vector_scores = [round(s, 3) for s in min_max_normalization(answer_distances, best, worst, reverse=False)]
    semantic = round(sum(vector_scores) / len(vector_scores) * 100, 2)
    return semantic, vector_scores


def answer_relevance_score(
    answer_semantic_score: float,
    answer_keyword_score: float,
    rerank_score: float,
    cfg: ScoringConfig = ScoringConfig(),
) -> float:
    """Port of the ARS assembly in ``apis.save_answer``.

    - ``answer_semantic_score``: 0-100 (from :func:`answer_vector_scores`).
    - ``answer_keyword_score``:  mean normalized BM25 vs answer, expressed 0-100.
    - ``rerank_score``:          mean question-side reranked score, 0-100.
    """
    raw = round(
        (answer_semantic_score / 100 * cfg.x_asem)
        + (answer_keyword_score / 100 * cfg.y_akey)
        + (rerank_score / 100 * cfg.z_arank),
        2,
    )
    ars = round(
        min_max_normalization([raw], cfg.ars_raw_max, cfg.ars_raw_min, reverse=False)[0] * 100,
        0,
    )
    return _clamp_0_100(ars)


# --------------------------------------------------------------------------- #
# Hallucination index (equation form)
# --------------------------------------------------------------------------- #
def hallucination_index_equation(
    question_score: float, answer_relevance: float, cfg: ScoringConfig = ScoringConfig()
) -> float:
    """Port of the HI-by-equation block in ``apis.save_answer``.

    ``HI_raw = 1 - (p*QRS + q*ARS)/(p+q)`` on 0-1 fractions, then rescaled from the
    [minHI, maxHI]=[0.2,0.8] window to 0-100 and clamped. Per MyGPT, ``question_score``
    is the question *semantic* score (``question.relevance_score``); pass the full QRS
    only if you intend to deviate from reference behaviour.
    """
    if question_score == 0:
        hi_raw = 1.0
    else:
        q_frac = question_score / 100
        a_frac = answer_relevance / 100
        denom = cfg.p_qrs + cfg.q_ars
        hi_raw = 1 - (cfg.p_qrs * q_frac) / denom - (cfg.q_ars * a_frac) / denom
    hi = round((hi_raw - cfg.hi_min) / (cfg.hi_max - cfg.hi_min) * 100, 0)
    return _clamp_0_100(hi)


# --------------------------------------------------------------------------- #
def _clamp_0_100(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 100:
        return 100.0
    return float(value)


# --------------------------------------------------------------------------- #
# Answer-side BM25 (Phase 6)
# --------------------------------------------------------------------------- #
# MyGPT indexes the retrieved contexts with bm25s (method "lucene", k1=1.5, b=0.75) and a
# PyStemmer stemmer for the documents' language, then scores the (translated) answer
# against them. Re-implemented in memory here (no bm25s temp-dir round trip). One
# deliberate deviation: the token pattern keeps Devanagari vowel signs / viramas inside
# words, which bm25s' \w\w+ splits on.
_TOKEN_RE = re.compile(r"[\w\u0900-\u097F]{2,}")
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


def answer_bm25_scores(
    answer: str, contexts: Sequence[str], language: str, k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Raw BM25 of the answer against each context (same order), contexts as the corpus."""
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
    """Per-source [0, 1] BM25 scores (min-max on bm25_best/worst) and their mean x100 —
    the ``keyword_score`` MyGPT feeds into ARS."""
    if not raw_scores:
        return 0.0, []
    scores = [
        round(v, 3)
        for v in min_max_normalization([round(r, 3) for r in raw_scores], cfg.bm25_best, cfg.bm25_worst)
    ]
    return round(sum(scores) / len(scores) * 100, 2), scores
