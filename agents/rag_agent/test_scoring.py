"""Golden-value tests for the MyGPT scoring port.

Expected values are computed by hand from the reference formulas (see scoring.py
source map) so a regression in the port is caught off-cluster. Run:

    python3 -m pytest agents/rag_agent/test_scoring.py      # or:
    python3 agents/rag_agent/test_scoring.py
"""

import math

from scoring import (
    EmbeddingCutoffs,
    ScoringConfig,
    answer_bm25_scores,
    answer_keyword_score,
    answer_relevance_score,
    bm25_tokenize,
    answer_vector_scores,
    hallucination_index_equation,
    min_max_normalization,
    question_relevance_score,
    question_semantic_score,
)

CUT = EmbeddingCutoffs(best_q=0.1, worst_q=1.0, best_ac=-1.0, worst_ac=1.0)
APPROX = 1e-6


def _close(a, b, tol=APPROX):
    return abs(a - b) <= tol


def test_min_max_forward_and_reverse():
    # forward: bigger is better -> worst maps to 0, best maps to 1
    assert _close(min_max_normalization([3.03], 3.03, -5.5)[0], 1.0)
    assert _close(min_max_normalization([-5.5], 3.03, -5.5)[0], 0.0)
    # reverse: smaller is better -> best(0.01) maps to 1, worst maps to 0
    assert _close(min_max_normalization([0.0, 2.0], 0.0, 2.0, reverse=True)[0], 1.0)
    assert _close(min_max_normalization([2.0], 0.0, 2.0, reverse=True)[0], 0.0)


def test_question_semantic_score():
    semantic, normalized = question_semantic_score([0.2, 0.4, 0.6], CUT)
    # buffered window: best=0.01, worst=1.27; mean=0.4 -> 69
    assert _close(semantic, 69.0)
    assert len(normalized) == 3
    # reverse-normalized best distance (0.2) is highest
    assert normalized[0] > normalized[2]


def test_question_semantic_zero_when_all_beyond_worst():
    # every distance beyond the buffered worst -> score collapses to 0
    semantic, _ = question_semantic_score([5.0, 6.0], CUT)
    assert _close(semantic, 0.0)


def test_question_relevance_score_qrs():
    # semantic=69, keyword=0.5, rerank=0.5, defaults a=4,b=-4,c=-1
    # raw = 2.76 - 2.0 - 0.5 = 0.26 ; QRS = (0.26+5.5)/8.53*100 = 67.53 -> 68
    assert _close(question_relevance_score(69.0, 0.5, 0.5), 68.0)


def test_answer_vector_scores():
    semantic, scores = answer_vector_scores([-0.2, 0.0, 0.2], CUT)
    # buffered ac window [-1.1, 1.1]; forward-normalized mean -> 50.0
    assert _close(semantic, 50.0)
    assert [round(s, 3) for s in scores] == [0.591, 0.5, 0.409]


def test_answer_relevance_score_ars():
    # answer_semantic=50, answer_keyword=40(0-100), rerank=30(0-100), x=5,y=-2,z=0
    # raw = 2.5 - 0.8 + 0 = 1.7 ; ARS = (1.7+6.42)/10.07*100 = 80.64 -> 81
    assert _close(answer_relevance_score(50.0, 40.0, 30.0), 81.0)


def test_hallucination_index_equation():
    # question_semantic=69, ARS=81, p=1,q=2
    # hi_raw = 1 - 0.69/3 - 2*0.81/3 = 0.23 ; HI = (0.23-0.2)/0.6*100 = 5
    assert _close(hallucination_index_equation(69.0, 81.0), 5.0)


def test_hallucination_index_clamps_when_question_score_zero():
    # question_score=0 -> hi_raw=1 -> (1-0.2)/0.6*100 = 133.3 -> clamped to 100
    assert _close(hallucination_index_equation(0.0, 50.0), 100.0)


def test_perfect_answer_has_low_hallucination():
    # strong QRS/ARS -> HI near 0 (clamped)
    assert hallucination_index_equation(100.0, 100.0) == 0.0


def test_config_overrides_change_qrs():
    strict = ScoringConfig(a_qsem=1.0, b_qkey=0.0, c_qrank=0.0)
    # raw = 69/100*1 = 0.69 ; QRS = (0.69+5.5)/8.53*100 = 72.57 -> 73
    assert _close(question_relevance_score(69.0, 0.5, 0.5, strict), 73.0)


# --- Phase 6: answer-side BM25 ------------------------------------------------ #
def test_bm25_lucene_golden():
    # 2 docs, no stemming ("unknown"): "kinase" in doc 0 only (df=1, N=2):
    # idf = ln(1 + 1.5/1.5) = ln 2; dl = avgdl = 2 -> norm = 1.5; tf=1 -> 1/2.5.
    raw = answer_bm25_scores("kinase", ["kinase binds", "other words"], "unknown")
    assert _close(raw[0], math.log(2) / 2.5) and raw[1] == 0.0


def test_bm25_stemming_per_language():
    # French Snowball folds "politiques" / "politique"; English doesn't stem French words
    # to the same root, so only the French stemmer matches.
    ctx = ["La politique nationale de santé", "Le budget annuel"]
    assert answer_bm25_scores("les politiques", ctx, "french")[0] > 0
    assert bm25_tokenize("Políticas sanitarias", "spanish") == bm25_tokenize("política sanitaria", "spanish")
    assert bm25_tokenize("kinases binding", "english") == ["kinas", "bind"]


def test_bm25_devanagari_tokens_stay_whole():
    # Vowel signs (matras) are Unicode Mn/Mc; they must not split the word.
    toks = bm25_tokenize("बच्चों में कैंसर", "unknown")
    assert toks == ["बच्चों", "में", "कैंसर"]
    assert answer_bm25_scores("कैंसर", ["बच्चों में कैंसर", "स्वास्थ्य नीति"], "hindi")[0] > 0


def test_answer_keyword_score_normalizes_on_bm25_window():
    cfg = ScoringConfig()  # bm25_best=20, bm25_worst=0
    mean100, per = answer_keyword_score([10.0, 0.0, 5.0], cfg)
    assert per == [0.5, 0.0, 0.25] and _close(mean100, 25.0)
    assert answer_keyword_score([], cfg) == (0.0, [])


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                passed += 1
                print(f"PASS {name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
