"""Unit tests for the Phase-2 eval scorers + dataset builder (no workspace needed).

    uv run --with "mlflow>=3.4" --with pytest pytest eval/
"""

from __future__ import annotations

import os
import sys

import pytest

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
    assert len(eval_datasets.build_records("bioasq", REPO)) == 163


def test_bioasq_records():
    recs = eval_datasets.build_records("bioasq", REPO)
    types = [r["tags"]["question_type"] for r in recs]
    assert types.count("yesno") == 76 and types.count("factoid") == 87
    for r in recs:
        ci, exp = r["inputs"]["custom_inputs"], r["expectations"]
        assert ci["corpus"] == "biomed" and ci["k"] == eval_datasets.K_BIOASQ and "focus" not in ci
        if r["tags"]["question_type"] == "yesno":
            assert exp["expected_label"] in ("yes", "no") and "expected_values" not in exp
        else:
            assert exp["expected_values"] and "expected_label" not in exp


# --- private SME gold (synthetic CSVs: no St. Jude values in git) ----------- #
def _write_csv(path, header: list[str], rows: list[list[str]]):
    import csv

    with open(path, "w", newline="", encoding="cp1252") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_policy_gold_flip_rule(tmp_path, monkeypatch):
    _write_csv(tmp_path / eval_datasets.POLICY_GOLD,
               ["qid", "document (Global policy)", "Document language", "question_keyword", "question",
                "answer_1", "Correctness for Answer 1 (Correct/Incorrect)"],
               [["1", "DocA", "French", "K1", "q", "Yes, it does.", "Correct"],
                ["2", "DocA", "French", "K2", "q", "No.", "Incorrect"],
                ["3", "DocA", "French", "K3", "q", "Yes, children.", "Incorrect"],
                ["4", "DocA", "French", "K4", "q", "No.", ""],
                ["5", "DocA", "French", "K5", "q", "The policy says…", "Correct"]])
    monkeypatch.setattr(eval_datasets, "private_dir", lambda _b: str(tmp_path))
    g = eval_datasets.policy_gold(REPO)
    assert g[("DocA", "K1")] == {"expected_label": "yes", "expected_response": "Yes, it does.", "mygpt_label": "correct"}
    assert g[("DocA", "K2")]["expected_label"] == "yes" and g[("DocA", "K2")]["expected_response"] is None
    assert g[("DocA", "K3")]["expected_label"] == "no"
    assert ("DocA", "K4") not in g and ("DocA", "K5") not in g  # unlabelled / no yes-no opening


def test_kinase_gold_realigns_shifted_block(tmp_path, monkeypatch):
    q = {"11": "Is there any information about K in this paper? ... what is the expression system for K ...",
         "15": "Is there any information about K in this paper? ... what assay was used to detect ...",
         "16": "Is there any information about K in this paper? ... in which figure or table is there ..."}
    header = ["id", "question_id", "pubmed_id", "kinase", "Question", "ground_truth",
              "answer_context_5", "label_context_5", "answer_context_15", "label_context_15",
              "answer_str_pre", "label_str_pre"]
    lab = ["a", "Correct", "a", "Correct", "a", "Correct"]
    rows = [
        # aligned pair P1
        ["1", "11", "P1", "K", q["11"], "HEK293"] + lab,
        ["2", "15", "P1", "K", q["15"], "protein kinase assay"] + lab,
        # P2: from its figure row on, question_id / ground_truth sit one row too high
        ["3", "11", "P2", "K", q["16"], "COS"] + lab,
        ["4", "15", "P3", "K", q["11"], "protein kinase assay"] + lab,
        ["5", "16", "P3", "K", q["15"], "Fig.4"] + lab,
        ["6", "11", "P3", "K", q["16"], ""] + lab,
        # back in step
        ["7", "15", "P4", "K", q["15"], "protein kinase assay"] + lab,
    ]
    _write_csv(tmp_path / eval_datasets.KINASE_GOLD, header, rows)
    monkeypatch.setattr(eval_datasets, "private_dir", lambda _b: str(tmp_path))
    g = eval_datasets.kinase_gold_rows(REPO)
    assert g[("P1", "K", "11")]["ground_truth"] == "HEK293" and not g[("P1", "K", "11")]["realigned"]
    assert g[("P2", "K", "17")]["ground_truth"] == ""            # block boundary: its gold was lost
    assert g[("P3", "K", "11")]["ground_truth"] == "COS" and g[("P3", "K", "11")]["realigned"]
    assert g[("P3", "K", "16v2")]["ground_truth"] == "protein kinase assay"
    assert g[("P3", "K", "17")]["ground_truth"] == "Fig.4"
    assert g[("P4", "K", "16v2")]["ground_truth"] == "protein kinase assay" and not g[("P4", "K", "16v2")]["realigned"]


def test_kinase_expectations_shape():
    assert eval_datasets.kinase_expectations("11", "E.coli;mammalian") == {"expected_values": ["E.coli", "mammalian"]}
    assert eval_datasets.kinase_expectations("17", "Fig.1;Fig.1;SFig.8.") == {"expected_values": ["Fig.1", "SFig.8"]}
    facts = eval_datasets.kinase_expectations("16v2", "protein kinase assay;protein kinase assay")["expected_facts"]
    assert len(facts) == 1 and "protein kinase assay" in facts[0]
    assert eval_datasets.kinase_expectations("11", "") == {}


_PRIVATE = eval_datasets.private_dir(REPO)


@pytest.mark.skipif(_PRIVATE is None, reason="St. Jude's private ground truths not present")
def test_private_gold_join_yields():
    for name in ("health_policies_fr", "health_policies_es"):
        recs = eval_datasets.build_records(name, REPO)
        gold = [r["tags"]["gold"] for r in recs]
        assert gold.count("sme") == 16 * 10 and gold.count("inherited") == 5 * 10
        assert all(r["expectations"].get("expected_label") in ("yes", "no") for r in recs)
        for r in recs:  # variants: label only
            if r["tags"]["gold"] == "inherited":
                assert r["expectations"]["gold_scope"] == "label_only" and "expected_response" not in r["expectations"]
    recs = eval_datasets.build_records("kinase_literature", REPO)
    assert len(eval_datasets.kinase_gold_rows(REPO)) == 282  # every (paper, kinase, question) once
    with_gold = [r for r in recs if r["tags"]["gold"] != "none"]
    assert len(with_gold) >= 180
    for r in with_gold:  # every gold value fits its question type
        qid, exp = r["tags"]["question_id"], r["expectations"]
        if qid == "16v2":
            assert exp["expected_facts"]
        else:
            fig = [eval_datasets._FIG_LIKE.match(v) is not None for v in exp["expected_values"]]
            assert all(fig) if qid == "17" else not any(fig)


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


# --- gold scorers ----------------------------------------------------------- #
def test_yes_no_correct():
    assert _val(s.yes_no_correct(outputs=_out("No, it does not."), expectations={"expected_label": "no"})) is True
    assert _val(s.yes_no_correct(outputs=_out("Oui, ..."), expectations={"expected_label": "no"})) is False
    assert _val(s.yes_no_correct(outputs=_out("Unclear."), expectations={"expected_label": "yes"})) is False
    assert s.yes_no_correct(outputs=_out("Yes"), expectations={"guidelines": ["x"]}) is None


def test_expression_system_correct():
    inputs = _kinase_inputs("11")
    exp = {"expected_values": ["HEK293", "E.coli"]}
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['HEK293T']"), expectations=exp)) is True
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['Escherichia coli']"), expectations=exp)) is True
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['COS-7']"), expectations=exp)) is False
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['Plat-E']"), expectations=exp)) is True
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['BL21(DE3)']"), expectations=exp)) is True
    insect = {"expected_values": ["SF21"]}
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['insect cell']"), expectations=insect)) is True
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("['Sf9']"), expectations=insect)) is True
    assert _val(s.expression_system_correct(inputs=inputs, outputs=_out("no answer"), expectations=exp)) is False
    assert s.expression_system_correct(inputs=inputs, outputs=_out("['HEK293']"), expectations={}) is None
    assert s.expression_system_correct(inputs=_kinase_inputs("17"), outputs=_out("x"), expectations=exp) is None


def test_figure_key():
    assert s.figure_key("Fig. 4B") == "fig4"
    assert s.figure_key("FIG.5 A") == "fig5"
    assert s.figure_key("Figure 1A") == "fig1"
    assert s.figure_key("Table III") == "table3"
    assert s.figure_key("SFig.8") == "sfig8"
    assert s.figure_key("Supplementary Figure S3") == "sfig3"
    assert s.figure_key("ST1") == "stable1"
    assert s.figure_key("Fig") is None and s.figure_key("the methods section") is None


def test_figure_ref_match():
    inputs = _kinase_inputs("17")
    exp = {"expected_values": ["Fig.1", "Fig.4"]}
    fb = {f.name: f.value for f in s.figure_ref_match(inputs=inputs, outputs=_out("['Fig. 1A', 'Fig. 2']"), expectations=exp)}
    assert fb == {"figure_ref_match": True, "figure_ref_recall": 0.5}
    fb = {f.name: f.value for f in s.figure_ref_match(inputs=inputs, outputs=_out("no answer"), expectations=exp)}
    assert fb == {"figure_ref_match": False, "figure_ref_recall": 0.0}
    assert s.figure_ref_match(inputs=inputs, outputs=_out("x"), expectations={}) is None


def test_factoid_match():
    exp = {"expected_values": ["EGFRvIII", "EGFR variant III"]}
    assert _val(s.factoid_match(outputs=_out("It is the EGFR variant III mutation."), expectations=exp)) is True
    assert _val(s.factoid_match(outputs=_out("It is BRAF."), expectations=exp)) is False
    assert s.factoid_match(outputs=_out("Yes"), expectations={"expected_label": "yes", "expected_values": ["x"]}) is None


def test_gold_correctness_skips_without_gold():
    assert s.gold_correctness(inputs={"input": []}, outputs=_out("x"), expectations={"guidelines": ["g"]}) is None
    # SJ Global variants inherit only the yes/no label, even if a stale expected_response survives
    stale = {"expected_label": "yes", "expected_response": "Yes, ...", "gold_scope": "label_only"}
    assert s.gold_correctness(inputs={"input": []}, outputs=_out("Yes"), expectations=stale) is None


def test_scorer_sets_build():
    names = {sc.name for sc in s.scorers_for("sj_global")}
    assert {"retrieval_groundedness", "yes_no_format", "answer_label", "mygpt_trust", "yes_no_correct", "gold_correctness"} <= names
    names = {sc.name for sc in s.scorers_for("bluesky", vocab=["E. coli"])}
    assert {"js_array_valid", "vocab_compliance", "kinase_in_context", "expression_system_correct", "figure_ref_match"} <= names
    names = {sc.name for sc in s.scorers_for("bioasq")}
    assert {"yes_no_correct", "factoid_match", "gold_correctness", "retrieval_ok"} <= names
    assert "yes_no_format" not in names and "focus_adherence" not in names


# --- run_eval post-run helpers ---------------------------------------------- #
import pandas as pd  # noqa: E402

import run_eval  # noqa: E402


def _persisted_row(record: dict, label: str, hi: float, grounded: str) -> dict:
    # Endpoint traces record the root-span input as {"request": <ResponsesAgentRequest>}.
    return {"trace_id": "t", "request": {"request": {**record["inputs"], "tool_choice": None}},
            "answer_label/value": label, "hi_eq/value": hi, "retrieval_groundedness/value": grounded}


def test_coding_matrix_unwraps_endpoint_request():
    records = eval_datasets.build_records("health_policies_fr", REPO)
    rows = pd.DataFrame([_persisted_row(records[0], "yes", 5, "yes"), _persisted_row(records[1], "no", 50, "no")])
    m = run_eval.coding_matrix(rows, records)
    doc = records[0]["inputs"]["custom_inputs"]["focus"]["documents"][0]
    assert m.loc[doc, records[0]["tags"]["question_key"]] == "yes"
    assert m.loc[doc, records[1]["tags"]["question_key"]] == "no"


def test_hi_vs_groundedness():
    records = eval_datasets.build_records("health_policies_fr", REPO)
    rows = pd.DataFrame([_persisted_row(records[0], "yes", 5, "yes"), _persisted_row(records[1], "no", 50, "no")])
    assert run_eval.hi_vs_groundedness(rows) == 1.0  # higher HI on the ungrounded row
    assert run_eval.auroc([1, 2], [True, True]) is None


def _gold_records():
    recs = eval_datasets.build_records("health_policies_fr", REPO)[:2]
    recs[0]["expectations"]["expected_label"], recs[0]["tags"]["gold"] = "yes", "sme"
    recs[1]["expectations"]["expected_label"], recs[1]["tags"]["gold"] = "yes", "inherited"
    return recs


def test_gold_breakdown_by_provenance():
    recs = _gold_records()
    rows = pd.DataFrame([
        {**_persisted_row(recs[0], "yes", 5, "yes"), "yes_no_correct/value": True},
        {**_persisted_row(recs[1], "no", 5, "yes"), "yes_no_correct/value": False},
    ])
    assert run_eval.gold_breakdown(rows, recs) == {"yes_no_correct/sme/mean": 1.0, "yes_no_correct/inherited/mean": 0.0}


def test_coding_matrix_vs_gold():
    recs = _gold_records()
    rows = pd.DataFrame([_persisted_row(recs[0], "yes", 5, "yes"), _persisted_row(recs[1], "no", 5, "yes")])
    m = run_eval.coding_matrix_vs_gold(rows, recs)
    assert list(m["correct"]) == [True, False] and list(m["gold_source"]) == ["sme", "inherited"]


# --- reference baseline ---------------------------------------------------- #
import reference_baseline as rb  # noqa: E402


def test_reference_summarize_agreement():
    rows = [{"use_case": "bluesky", "config": "ctx5", "sme": True},
            {"use_case": "bluesky", "config": "ctx5", "sme": False},
            {"use_case": "bluesky", "config": "ctx5", "sme": True}]
    scored = [{"expression_system_correct": True}, {"expression_system_correct": True}, {}]
    m = rb.summarize(rows, scored)
    assert m["mygpt_ref/bluesky/ctx5/sme_accuracy"] == 2 / 3
    assert m["mygpt_ref/bluesky/ctx5/expression_system_correct"] == 1.0
    assert m["mygpt_ref/bluesky/ctx5/expression_system_correct/agreement"] == 0.5


def test_reference_summarize_splits_gold_provenance():
    rows = [{"use_case": "bluesky", "config": "ctx5", "sme": True, "gold": "sme"},
            {"use_case": "bluesky", "config": "ctx5", "sme": True, "gold": "sme_realigned"}]
    scored = [{"expression_system_correct": True}, {"expression_system_correct": False}]
    m = rb.summarize(rows, scored)
    assert m["mygpt_ref/bluesky/ctx5/sme/expression_system_correct/agreement"] == 1.0
    assert m["mygpt_ref/bluesky/ctx5/sme_realigned/expression_system_correct/agreement"] == 0.0
    assert m["mygpt_ref/bluesky/ctx5/expression_system_correct/agreement"] == 0.5


def test_reference_bioasq_rows_score_without_judge():
    rows = [r for r in rb.reference_rows(REPO) if r["use_case"] == "bioasq"]
    assert len(rows) == 163
    scored = [rb.score_row(r, judge=False) for r in rows]
    yesno = [v["yes_no_correct"] for r, v in zip(rows, scored) if r["question_type"] == "yesno"]
    assert len(yesno) == 76 and sum(map(bool, yesno)) > 50  # MyGPT's published answers are mostly right
    assert all("factoid_match" in v for r, v in zip(rows, scored) if r["question_type"] == "factoid")


@pytest.mark.skipif(_PRIVATE is None, reason="St. Jude's private ground truths not present")
def test_reference_kinase_rows():
    rows = [r for r in rb.reference_rows(REPO) if r["use_case"] == "bluesky"]
    assert len(rows) == 3 * 282
    scored = [rb.score_row(r, judge=False) for r in rows]
    m = rb.summarize(rows, scored)
    assert 0 < m["mygpt_ref/bluesky/ctx15/sme_accuracy"] <= 1
    assert "mygpt_ref/bluesky/ctx15/expression_system_correct/agreement" in m


def test_bioasq_gates_skip_focus_adherence():
    gates = os.path.join(REPO, "eval", "gates.yaml")
    def focus_failures(use_case):
        return [f for f in run_eval.check_gates({}, use_case, gates) if f.startswith("focus_adherence")]

    assert focus_failures("bioasq") == []
    assert focus_failures("sj_global")  # still enforced there
