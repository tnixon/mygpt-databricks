"""Tests for the Phase-4 pre-retrieved sources path (custom_inputs.sources -> _retrieve shape).

    uv run --no-project --with-requirements eval/requirements.txt pytest -q agents/rag_agent/test_provided_sources.py
"""

from agent import MAX_PROVIDED_CHARS, sources_from_inputs


def test_maps_to_retrieve_shape():
    out = sources_from_inputs(
        [{"chunk_id": "c1", "source_name": "p.pdf", "corpus": "upload:x", "section": "Methods", "page": 3,
          "chunk_text": "Imatinib binds ABL1.", "similarity": 0.81234, "bm25_score": 0.5}],
        k=5,
    )
    assert out == [{
        "chunk_id": "c1", "source_name": "p.pdf", "corpus": "upload:x", "section": "Methods", "page": 3,
        "context": "Imatinib binds ABL1.", "similarity": 0.8123, "vector_distance_raw": 0.1877,
        "bm25_score": 0.5, "rank": 1,
    }]


def test_caps_drops_and_clamps():
    raw = [{"chunk_text": f"t{i}", "similarity": 2, "bm25_score": -1, "page": "x"} for i in range(20)]
    raw.insert(0, {"chunk_text": "   "})  # empty text: dropped
    raw.insert(0, "not a dict")
    out = sources_from_inputs(raw, k=50)
    assert len(out) == 10  # hard cap
    assert out[0]["similarity"] == 1.0 and out[0]["vector_distance_raw"] == 0.0
    assert out[0]["bm25_score"] == 0.0 and out[0]["page"] is None
    assert [s["rank"] for s in out] == list(range(1, 11))
    assert len(sources_from_inputs(raw, k=3)) == 3


def test_truncates_text_and_rejects_non_lists():
    out = sources_from_inputs([{"chunk_text": "x" * (MAX_PROVIDED_CHARS + 10), "similarity": float("nan")}], k=1)
    assert len(out[0]["context"]) == MAX_PROVIDED_CHARS
    assert out[0]["similarity"] == -1.0
    assert sources_from_inputs({"chunk_text": "x"}, k=1) == []
    assert sources_from_inputs(None, k=1) == []
