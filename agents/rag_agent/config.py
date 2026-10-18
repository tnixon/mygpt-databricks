"""Load per-dataset scoring config from the Delta tables, with safe fallbacks.

The ResponsesAgent reads tunable coefficients, per-model distance cutoffs, and
reranker thresholds that the ETL job seeds into Unity Catalog
(`structural_biology.mygpt.{dataset_config,embedding_model_cutoffs,reranker_config}`).

Config is loaded once at endpoint init via the Databricks SQL Statement Execution
API (no Spark on a serving endpoint). If a warehouse/table is unavailable, we fall
back to the packaged MyGPT defaults so the endpoint still serves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from scoring import EmbeddingCutoffs, ScoringConfig

CATALOG_SCHEMA = os.environ.get("MYGPT_SCHEMA", "structural_biology.mygpt")

# Bootstrap cutoffs for bge-m3 in cosine-DISTANCE space (= 1 - cosine similarity).
# Mirrors etl/config/embedding_model_cutoffs.csv; recalibrate per workstream C.
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
    translate_answer: bool = False
    answer_bm25: bool = False


RERANK_MODEL = "cross-encoder/qnli-electra-base"
RERANK_MODEL_MULTILINGUAL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CUTOFFS = {RERANK_MODEL: 0.10, RERANK_MODEL_MULTILINGUAL: 0.05}
TRANSLATABLE = {"english", "french", "spanish", "portuguese", "german", "italian", "dutch", "arabic", "hindi", "hungarian"}

# Governed corpora whose documents aren't English (mirror of etl/config/dataset_config.csv).
# Upload-library config rows are named default_<language>. Packaged defaults derive from
# the language, so the endpoint applies them even without a warehouse: non-English uses
# MyGPT's multilingual mmarco reranker and query/answer translation (NLI is English-only).
CORPUS_LANGUAGE = {
    "health_policies_fr": "french", "health_policies_es": "spanish", "health_policies_hi": "hindi",
}


def dataset_language(dataset_name: str) -> str:
    if dataset_name.startswith("default_"):
        return dataset_name[len("default_"):]
    return CORPUS_LANGUAGE.get(dataset_name, "english")


def _fallback(dataset_name: str) -> DatasetConfig:
    language = dataset_language(dataset_name)
    reranker = RERANK_MODEL if language == "english" else RERANK_MODEL_MULTILINGUAL
    return DatasetConfig(
        dataset_name=dataset_name,
        embedding_model="bge-m3",
        reranker=reranker,
        use_bm25=True,
        use_reranker=True,
        documents_language=language,
        hi_by_equation=True,
        scoring=ScoringConfig(),
        cutoffs=_DEFAULT_CUTOFFS,
        reranker_cutoff=RERANK_CUTOFFS[reranker],
        translate_query=language in TRANSLATABLE and language != "english",
        translate_answer=False,
        answer_bm25=False,
    )


def load_dataset_config(dataset_name: str = "mygpt") -> DatasetConfig:
    """Return config for a dataset, reading Delta if a warehouse is configured."""
    warehouse_id = os.environ.get("MYGPT_WAREHOUSE_ID")
    if not warehouse_id:
        return _fallback(dataset_name)
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()

        def q(sql: str) -> list[dict]:
            resp = w.statement_execution.execute_statement(
                warehouse_id=warehouse_id, statement=sql, wait_timeout="30s"
            )
            cols = [c.name for c in resp.manifest.schema.columns]
            rows = resp.result.data_array or []
            return [dict(zip(cols, r)) for r in rows]

        ds_rows = q(
            f"SELECT * FROM {CATALOG_SCHEMA}.dataset_config WHERE dataset_name = '{dataset_name}'"
        )
        if not ds_rows:
            return _fallback(dataset_name)
        d = ds_rows[0]

        cut_rows = q(
            f"SELECT * FROM {CATALOG_SCHEMA}.embedding_model_cutoffs "
            f"WHERE model_name = '{d['embedding_model']}'"
        )
        cut = cut_rows[0] if cut_rows else None

        rk_rows = q(
            f"SELECT cutoff_score FROM {CATALOG_SCHEMA}.reranker_config "
            f"WHERE model_name = '{d['reranker']}'"
        )

        return DatasetConfig(
            dataset_name=dataset_name,
            embedding_model=str(d["embedding_model"]),
            reranker=str(d["reranker"]),
            use_bm25=_as_bool(d["use_bm25"]),
            use_reranker=_as_bool(d["use_reranker"]),
            documents_language=str(d["documents_language"]),
            hi_by_equation=_as_bool(d["HI_by_equation"]),
            scoring=ScoringConfig(
                a_qsem=float(d["coefficient_a_Qsem"]),
                b_qkey=float(d["coefficient_b_Qkey"]),
                c_qrank=float(d["coefficient_c_Qrank"]),
                x_asem=float(d["coefficient_x_Asem"]),
                y_akey=float(d["coefficient_y_Akey"]),
                z_arank=float(d["coefficient_z_Arank"]),
                p_qrs=float(d["coefficient_p_QRS"]),
                q_ars=float(d["coefficient_q_ARS"]),
            ),
            cutoffs=EmbeddingCutoffs(
                best_q=float(cut["best_distance_q"]) if cut else _DEFAULT_CUTOFFS.best_q,
                worst_q=float(cut["worst_distance_q"]) if cut else _DEFAULT_CUTOFFS.worst_q,
                best_ac=float(cut["best_distance_ac"]) if cut else _DEFAULT_CUTOFFS.best_ac,
                worst_ac=float(cut["worst_distance_ac"]) if cut else _DEFAULT_CUTOFFS.worst_ac,
            ),
            reranker_cutoff=float(rk_rows[0]["cutoff_score"]) if rk_rows else RERANK_CUTOFFS.get(str(d["reranker"]), 0.10),
            translate_query=_as_bool(d.get("translate_query", False)),
            translate_answer=_as_bool(d.get("translate_answer", False)),
            answer_bm25=_as_bool(d.get("answer_bm25", False)),
        )
    except Exception as exc:  # noqa: BLE001 - never let config fail the endpoint
        print(f"[config] falling back to defaults: {type(exc).__name__}: {exc}")
        return _fallback(dataset_name)


def _as_bool(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "t", "yes")
