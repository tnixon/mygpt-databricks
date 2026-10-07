"""Random-forest Hallucination-Index variant (MyGPT alternative to the equation).

Port of ``HI_prediction_by_random_forest.predict_hallucination_index`` adapted to
operate on plain dict sources (the agent's shape) rather than Django ORM objects.
The pickled artifact (``random_forest_hybrid_model_base.pkl``, a dict with keys
``model`` / ``feature_names`` / ``best_cutoff``) is shipped as an MLflow artifact
with the agent and loaded lazily.

Phase-1 treats this as a *best-effort toggle*: if the pickle cannot be loaded under
the serving image's scikit-learn (version drift) the agent falls back to the
equation HI and logs the reason. Retraining it on Databricks as a first-class,
version-pinned output is the documented higher-parity step.

Feature order (must match ``artifact['feature_names']``):
  vector_scores_answer_{mean,max,min,std}, bm25_scores_answer_{mean,max,min,std},
  vector_scores_question_{mean,max,min,std}, bm25_scores_question_{mean,max,min,std},
  reranked_score_question_{mean,max,min,std},
  rerank_sentiments_answer_contradiction_count, _entailment_count
(interleaved exactly as the reference assembles them — see ``_feature_vector``).
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Sequence

import numpy as np

_ARTIFACT: dict[str, Any] | None = None
_LOAD_ERROR: str | None = None

# Default lookup paths; the agent sets MYGPT_RF_MODEL_PATH to the resolved
# MLflow artifact path at init.
_CANDIDATE_PATHS = [
    os.environ.get("MYGPT_RF_MODEL_PATH", ""),
    os.path.join(os.path.dirname(__file__), "artifacts", "random_forest_hybrid_model_base.pkl"),
]


def _load_artifact() -> dict[str, Any] | None:
    global _ARTIFACT, _LOAD_ERROR
    if _ARTIFACT is not None or _LOAD_ERROR is not None:
        return _ARTIFACT
    for path in _CANDIDATE_PATHS:
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as fh:
                    _ARTIFACT = pickle.load(fh)
                return _ARTIFACT
            except Exception as exc:  # noqa: BLE001 - version drift etc.
                _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
                return None
    _LOAD_ERROR = "random_forest_hybrid_model_base.pkl not found"
    return None


def _safe_stats(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Port of ``HI_prediction_by_random_forest._safe_stats``."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    return float(np.mean(arr)), float(np.max(arr)), float(np.min(arr)), float(np.std(arr))


def _feature_vector(
    vector_scores_answer: Sequence[float],
    bm25_scores_answer: Sequence[float],
    vector_scores_question: Sequence[float],
    bm25_scores_question: Sequence[float],
    reranked_scores_question: Sequence[float],
    rerank_sentiments: Sequence[Any],
) -> list[float]:
    a_v_mean, a_v_max, a_v_min, a_v_std = _safe_stats(vector_scores_answer)
    a_b_mean, a_b_max, a_b_min, a_b_std = _safe_stats(bm25_scores_answer)
    q_v_mean, q_v_max, q_v_min, q_v_std = _safe_stats(vector_scores_question)
    q_b_mean, q_b_max, q_b_min, q_b_std = _safe_stats(bm25_scores_question)
    q_r_mean, q_r_max, q_r_min, q_r_std = _safe_stats(reranked_scores_question)

    contradiction_count = 0
    entailment_count = 0
    for sentiment in rerank_sentiments:
        label = str(sentiment[0] if isinstance(sentiment, (list, tuple)) and sentiment else sentiment).lower()
        if label == "contradiction":
            contradiction_count += 1
        elif label == "entailment":
            entailment_count += 1

    return [
        a_v_mean, a_b_mean, q_v_mean, q_b_mean, q_r_mean,
        a_v_max, a_b_max, a_v_min, a_b_min, a_v_std, a_b_std,
        q_v_max, q_b_max, q_r_max, q_v_min, q_b_min, q_r_min,
        q_v_std, q_b_std, q_r_std,
        float(contradiction_count), float(entailment_count),
    ]


def predict_hallucination_index(
    vector_scores_answer: Sequence[float],
    bm25_scores_answer: Sequence[float],
    sources: Sequence[dict],
    rerank_sentiments: Sequence[Any],
) -> float | None:
    """Return the RF incorrectness probability (0-100), or ``None`` if unavailable.

    ``sources`` are the question-side reranked sources (dicts carrying
    ``vector_score`` / ``bm25_score`` / ``rerank_score`` keys).
    """
    artifact = _load_artifact()
    if artifact is None:
        return None

    vector_scores_question = [float(s["vector_score"]) for s in sources if s.get("vector_score") is not None]
    bm25_scores_question = [float(s["bm25_score"]) for s in sources if s.get("bm25_score") is not None]
    reranked_scores_question = [float(s["rerank_score"]) for s in sources if s.get("rerank_score") is not None]

    features = _feature_vector(
        vector_scores_answer,
        bm25_scores_answer,
        vector_scores_question,
        bm25_scores_question,
        reranked_scores_question,
        rerank_sentiments,
    )
    try:
        model = artifact["model"]
        proba = model.predict_proba(np.array([features], dtype=float))[:, 1]
        return round(float(proba[0]) * 100, 0) if len(proba) else None
    except Exception:  # noqa: BLE001
        return None


def load_error() -> str | None:
    """Diagnostic: why the RF artifact is unavailable, if it is."""
    _load_artifact()
    return _LOAD_ERROR
