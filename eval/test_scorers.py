"""Unit tests for the Phase-2 eval scorers + dataset builder (no workspace needed).

    uv run --with "mlflow>=3.4" --with pytest pytest eval/
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_datasets  # noqa: E402
import scorers as s  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _out(text: str, sources=None, **co) -> dict:
    return {
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
        "custom_outputs": {"sources": sources if sources is not None else [{"source_name": "a.pdf", "snippet": ""}], **co},
    }


def _kinase_inputs(qid: str, kinase: str = "EPHA8", doc: str = "10498895.pdf") -> dict:
    rec = next(
        r for r in eval_datasets.kinase_records(REPO, vocab=[])
        if r["tags"]["question_id"] == qid and r["tags"]["kinase"] == kinase
    )
    rec["inputs"]["custom_inputs"]["focus"]["documents"] = [doc]
    return rec["inputs"]


def _val(fb):
    return fb.value if hasattr(fb, "value") else fb


# --- dataset builder ------------------------------------------------------- #
def test_dataset_row_counts():
    assert len(eval_datasets.build_records("health_policies_fr", REPO)) == 21 * 10
    assert len(eval_datasets.build_records("health_policies_es", REPO)) == 21 * 10
    assert len(eval_datasets.build_records("kinase_literature", REPO)) == 3 * 94


def test_records_are_agent_requests():
    rec = eval_datasets.build_records("health_policies_fr", REPO)[0]
    ci = rec["inputs"]["custom_inputs"]
    assert rec["inputs"]["input"][0]["role"] == "user"
    assert ci["corpus"] == "health_policies_fr" and ci["k"] == eval_datasets.K
    assert ci["focus"]["documents"][0].endswith(".pdf")
    assert rec["expectations"]["guidelines"]


def test_kinase_question_roundtrip():
    for qid in ("11", "16v2", "17"):
        got_qid, kinase, names = s.kinase_question(_kinase_inputs(qid))
        assert got_qid == qid and kinase == "EPHA8" and "EEK" in names


def test_kinase_vocab_in_instructions():
    rec = eval_datasets.kinase_records(REPO, vocab=["E. coli", "Sf9 insect cells"])[0]
    assert "E. coli, Sf9 insect cells" in rec["inputs"]["custom_inputs"]["instructions"]
    assert rec["tags"]["vocab_pending"] == "false"


# --- shared ----------------------------------------------------------------- #
def test_retrieval_ok():
    assert _val(s.retrieval_ok(outputs=_out("x"))) is True
    assert _val(s.retrieval_ok(outputs=_out("x", sources=[]))) is False
    assert _val(s.retrieval_ok(outputs=_out("x", sources=[], retrieve_error="PermissionDenied"))) is False


def test_mygpt_trust():
    fbs = s.mygpt_trust(outputs=_out("x", qrs=54, ars=86, hi_eq=0, hi_rf=None))
    assert {f.name: f.value for f in fbs} == {"qrs": 54.0, "ars": 86.0, "hi_eq": 0.0}


def test_focus_adherence():
    inputs = {"input": [], "custom_inputs": {"focus": {"documents": ["a.pdf"]}}}
    assert s.focus_adherence(inputs=inputs, outputs=_out("x")) is True
    assert s.focus_adherence(inputs=inputs, outputs=_out("x", sources=[{"source_name": "b.pdf"}])) is False
    assert s.focus_adherence(inputs={"custom_inputs": {}}, outputs=_out("x")) is None


# --- SJ Global -------------------------------------------------------------- #
def test_yes_no():
    assert s.yes_no_label("Yes, the policy mentions children [1].") == "yes"
    assert s.yes_no_label("**No.** The document does not.") == "no"
    assert s.yes_no_label("Oui, le document...") == "yes"
    assert s.yes_no_label("The policy describes...") == "unparseable"
    assert _val(s.yes_no_format(outputs=_out("No."))) is True
    assert s.answer_label(outputs=_out("Yes — see [2]")) == "yes"


# --- BlueSky ---------------------------------------------------------------- #
def test_js_array_parsing():
    assert s.parse_js_array("['E. coli']") == ["E. coli"]
    assert s.parse_js_array('```javascript\n["Western blot", "kinase assay"]\n```') == ["Western blot", "kinase assay"]
    assert s.parse_js_array("E. coli") is None
    assert s.is_sentinel("no match.") and s.is_sentinel('["no answer"]') and not s.is_sentinel("['Sf9']")


def test_js_array_valid():
    inputs = _kinase_inputs("16v2")
    assert _val(s.js_array_valid(inputs=inputs, outputs=_out("['in vitro kinase assay']"))) is True
    assert _val(s.js_array_valid(inputs=inputs, outputs=_out("no match"))) is True
    assert _val(s.js_array_valid(inputs=inputs, outputs=_out("They used a kinase assay."))) is False
    assert s.js_array_valid(inputs={"input": [{"role": "user", "content": "hi"}]}, outputs=_out("x")) is None


def test_vocab_compliance():
    vc = s.make_vocab_compliance(["E. coli", "Sf9 insect cells"])
    assert _val(vc(inputs=_kinase_inputs("11"), outputs=_out("['e. coli']"))) is True
    assert _val(vc(inputs=_kinase_inputs("11"), outputs=_out("['HEK293']"))) is False
    assert vc(inputs=_kinase_inputs("17"), outputs=_out("['Fig. 2']")) is None


def test_figure_ref_format():
    inputs = _kinase_inputs("17")
    assert _val(s.figure_ref_format(inputs=inputs, outputs=_out("['Fig. 2A', 'Table 1', 'Supplementary Figure S3']"))) is True
    assert _val(s.figure_ref_format(inputs=inputs, outputs=_out("['the methods section']"))) is False


def test_kinase_in_context():
    inputs = _kinase_inputs("11")
    hit = _out("['E. coli']", sources=[{"source_name": "10498895.pdf", "snippet": "EEK was expressed in E. coli"}])
    miss = _out("['E. coli']", sources=[{"source_name": "10498895.pdf", "snippet": "unrelated text"}])
    fb = {f.name: f.value for f in s.kinase_in_context(inputs=inputs, outputs=hit, trace=None)}
    assert fb == {"kinase_in_context": True, "no_match_consistent": True}
    fb = {f.name: f.value for f in s.kinase_in_context(inputs=inputs, outputs=miss, trace=None)}
    assert fb == {"kinase_in_context": False, "no_match_consistent": False}
    fb = {f.name: f.value for f in s.kinase_in_context(inputs=inputs, outputs=_out("no match", sources=miss["custom_outputs"]["sources"]), trace=None)}
    assert fb["no_match_consistent"] is True


def test_scorer_sets_build():
    names = {sc.name for sc in s.scorers_for("sj_global")}
    assert {"retrieval_groundedness", "yes_no_format", "answer_label", "mygpt_trust"} <= names
    names = {sc.name for sc in s.scorers_for("bluesky", vocab=["E. coli"])}
    assert {"js_array_valid", "vocab_compliance", "kinase_in_context"} <= names
