"""BAAI/bge-m3 as an OpenAI-compatible embeddings model for Model Serving.

Logged with MLflow and deployed to the `mygpt-bge-m3` endpoint (embeddings task),
which the AI Search Delta Sync index uses as its MANAGED embedding model and which
the ResponsesAgent calls to embed queries/answers. bge-m3 is prefix-free for dense
retrieval, so documents and queries are embedded identically (no asymmetric
`search_document:` / `search_query:` handling — the reason we chose it over nomic).

bge-m3 produces 1024-dim dense vectors. Keep the vector dimension consistent with
the Vector Search index when (re)creating it.
"""

from __future__ import annotations

import mlflow
import pandas as pd


class BgeM3Embedding(mlflow.pyfunc.PythonModel):
    def load_context(self, context) -> None:  # noqa: ANN001
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("BAAI/bge-m3")

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def predict(self, context, model_input, params=None):  # noqa: ANN001
        # Accept a DataFrame/dict/list; return the embeddings-task response shape.
        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, dict):
            raw = model_input.get("input", model_input.get("inputs", []))
            texts = [raw] if isinstance(raw, str) else list(raw)
        elif isinstance(model_input, str):
            texts = [model_input]
        else:
            texts = list(model_input)

        embeddings = self._encode(texts)
        return {
            "object": "list",
            "data": [
                {"object": "embedding", "index": i, "embedding": e}
                for i, e in enumerate(embeddings)
            ],
            "model": "bge-m3",
        }
