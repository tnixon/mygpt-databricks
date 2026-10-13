"""Scorers for the MyGPT eval harness: guidelines judges + code scorers, plus gold
scorers that read ``expectations`` (expected_label / expected_values / expected_response /
expected_facts — see eval_datasets.py) and skip rows without gold.

``outputs`` is the ``mygpt-rag-agent`` ResponsesAgent response
(``{"output": [...message...], "custom_outputs": {qrs, ars, hi_eq, sources, ...}}``) and
``inputs`` the request body (``{"input": [...], "custom_inputs": {corpus, focus, ...}}``).
Scorers return ``None`` (an empty assessment) on rows they don't apply to.

``retrieval_ok`` and ``mygpt_trust`` are self-contained (imports + logic inside the
function body) so ``monitor.py`` can register them for production monitoring — MLflow
serializes a registered scorer's source, so it can't reference module helpers.
"""

from __future__ import annotations

import json
import re

from eval_datasets import yes_no_label
from mlflow.entities import Feedback
from mlflow.genai.scorers import (
    ExpectationsGuidelines,
    Guidelines,
    RelevanceToQuery,
    RetrievalGroundedness,
    RetrievalRelevance,
    Safety,
    scorer,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def answer_text(outputs) -> str:
    """Final assistant text from a ResponsesAgent response (dict) or a bare string."""
    if isinstance(outputs, str):
        return outputs
    parts = []
    for item in (outputs or {}).get("output", []) or []:
        if item.get("type") == "message":
            parts += [c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text"]
    return "".join(parts).strip()


def _question(inputs) -> str:
    for msg in reversed((inputs or {}).get("input", []) or []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


def _custom_outputs(outputs) -> dict:
    return (outputs or {}).get("custom_outputs") or {} if isinstance(outputs, dict) else {}


_SENTINELS = {"no match", "no answer"}


def _strip(text: str) -> str:
    """Drop markdown code fences / surrounding punctuation the LLM may add."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    return text.strip().strip(".").strip()


def parse_js_array(text: str) -> list | None:
    """Parse a JavaScript/JSON array answer (single or double quoted); None if not one."""
    text = _strip(text)
    if not (text.startswith("[") and text.endswith("]")):
        return None
    for candidate in (text, re.sub(r"'", '"', text)):
        try:
            value = json.loads(candidate)
            return value if isinstance(value, list) else None
        except json.JSONDecodeError:
            continue
    return None


def is_sentinel(text: str) -> bool:
    text = _strip(text).strip("'\"").lower()
    arr = parse_js_array(text)
    if arr is not None and len(arr) == 1 and isinstance(arr[0], str):
        text = arr[0].strip().strip(".").lower()
    return text in _SENTINELS


def kinase_question(inputs) -> tuple[str | None, str | None, list[str]]:
    """(question_id, kinase, names) from a kinase_literature request. The question type
    and kinase are recovered from the templated question text built in eval_datasets.py."""
    q = _question(inputs)
    m = re.match(r"Is there any information about (.+?) in this paper\?", q)
    if not m:
        return None, None, []
    kinase = m.group(1)
    if "what is the expression system" in q:
        qid = "11"
    elif "what assay was used" in q:
        qid = "16v2"
    elif "which figure or table" in q:
        qid = "17"
    else:
        qid = None
    names = [kinase]
    syn = re.search(r"is also known as (.+)\.$", q)
    if syn:
        names += [s.strip() for s in syn.group(1).split(",") if s.strip()]
    return qid, kinase, names


# --------------------------------------------------------------------------- #
# shared (all corpora) — also registered for production monitoring
# --------------------------------------------------------------------------- #
@scorer
def retrieval_ok(outputs) -> Feedback:
    """The agent retrieved at least one source and raised no retrieval error."""
    from mlflow.entities import Feedback

    co = (outputs or {}).get("custom_outputs") or {} if isinstance(outputs, dict) else {}
    err = co.get("retrieve_error")
    n = len(co.get("sources") or [])
    if err:
        return Feedback(value=False, rationale=f"retrieve_error: {err}")
    return Feedback(value=n > 0, rationale=f"{n} sources retrieved")


@scorer
def mygpt_trust(outputs) -> list:
    """MyGPT's own trust metrics (QRS / ARS / HI) as numeric feedback, so they aggregate
    per run and can be compared against the judges."""
    from mlflow.entities import Feedback

    co = (outputs or {}).get("custom_outputs") or {} if isinstance(outputs, dict) else {}
    out = []
    for key in ("qrs", "ars", "hi_eq", "hi_rf"):
        if co.get(key) is not None:
            out.append(Feedback(name=key, value=float(co[key])))
    return out


@scorer
def focus_adherence(inputs, outputs) -> bool | None:
    """Every source comes from the focused document(s)."""
    docs = set(((inputs or {}).get("custom_inputs") or {}).get("focus", {}).get("documents") or [])
    sources = _custom_outputs(outputs).get("sources") or []
    if not docs or not sources:
        return None
    return all(s.get("source_name") in docs for s in sources)


# --------------------------------------------------------------------------- #
# SJ Global — health_policies
# --------------------------------------------------------------------------- #
@scorer
def yes_no_format(outputs) -> Feedback:
    """The answer starts with Yes / No, as every SJ Global question instructs."""
    text = answer_text(outputs)
    label = yes_no_label(text)
    return Feedback(value=label != "unparseable", rationale=f"starts: {text[:40]!r}")


@scorer
def answer_label(outputs) -> str:
    """Categorical yes / no / unparseable — the cell value of the SJ Global
    policy x question coding matrix (see run_eval.coding_matrix)."""
    return yes_no_label(answer_text(outputs))


# --------------------------------------------------------------------------- #
# BlueSky — kinase_literature
# --------------------------------------------------------------------------- #
@scorer
def js_array_valid(inputs, outputs) -> Feedback | None:
    """Structured-output validity: a JS/JSON array of strings, or the 'no match' /
    'no answer' sentinel the prompts allow."""
    if kinase_question(inputs)[0] is None:
        return None
    text = answer_text(outputs)
    if is_sentinel(text):
        return Feedback(value=True, rationale="sentinel")
    arr = parse_js_array(text)
    ok = arr is not None and len(arr) > 0 and all(isinstance(x, str) and x.strip() for x in arr)
    return Feedback(value=ok, rationale=f"answer: {text[:80]!r}")


def make_vocab_compliance(vocab: list[str]):
    """Q11: the single extracted expression system is one of St. Jude's vocabulary
    entries. Built per run, as the vocabulary is data (expression_systems_list.csv)."""
    allowed = {v.strip().lower() for v in vocab}

    @scorer(name="vocab_compliance")
    def vocab_compliance(inputs, outputs) -> Feedback | None:
        if kinase_question(inputs)[0] != "11":
            return None
        text = answer_text(outputs)
        if is_sentinel(text):
            return Feedback(value=True, rationale="sentinel")
        arr = parse_js_array(text) or []
        ok = len(arr) == 1 and str(arr[0]).strip().lower() in allowed
        return Feedback(value=ok, rationale=f"extracted: {arr!r}")

    return vocab_compliance


_FIG_REF = re.compile(r"^(supplementa(ry|l)\s+)?(fig(ure)?s?\.?|tables?|s\d)\s*[a-z]?\d", re.IGNORECASE)


@scorer
def figure_ref_format(inputs, outputs) -> Feedback | None:
    """Q17: every element is a figure / table reference (Fig. 2A, Table 1, Supplementary Figure S3)."""
    if kinase_question(inputs)[0] != "17":
        return None
    text = answer_text(outputs)
    if is_sentinel(text):
        return Feedback(value=True, rationale="sentinel")
    arr = parse_js_array(text) or []
    ok = bool(arr) and all(_FIG_REF.match(str(x).strip()) for x in arr)
    return Feedback(value=ok, rationale=f"refs: {arr!r}")


@scorer
def kinase_in_context(inputs, outputs, trace) -> list | None:
    """Retrieval sanity for BlueSky: did the retrieved chunks mention the kinase (or a
    synonym)? And is a 'no match' answer consistent with that?"""
    qid, kinase, names = kinase_question(inputs)
    if qid is None:
        return None
    chunks = []
    if trace is not None:
        for span in trace.search_spans(span_type="RETRIEVER"):
            chunks += [str(d.get("page_content", "")) for d in (span.outputs or []) if isinstance(d, dict)]
    if not chunks:  # fall back to the snippets returned in custom_outputs
        chunks = [str(s.get("snippet", "")) for s in _custom_outputs(outputs).get("sources") or []]
    blob = " ".join(chunks).lower()
    hits = [n for n in names if len(n) > 2 and re.search(rf"\b{re.escape(n.lower())}\b", blob)]
    mentioned = bool(hits)
    sentinel = is_sentinel(answer_text(outputs))
    return [
        Feedback(name="kinase_in_context", value=mentioned, rationale=f"matched: {hits[:5]}"),
        # An extracted value when the kinase never appears in the retrieved context is a
        # likely hallucination; a sentinel is always acceptable ('no answer' = the paper
        # mentions the kinase but not the asked detail).
        Feedback(
            name="no_match_consistent", value=mentioned or sentinel,
            rationale=f"kinase {'found' if mentioned else 'absent'}; answer {'is' if sentinel else 'is not'} a sentinel",
        ),
    ]


# --------------------------------------------------------------------------- #
# gold scorers — read expectations; None on rows without gold
# --------------------------------------------------------------------------- #
@scorer
def yes_no_correct(outputs, expectations) -> Feedback | None:
    """SJ Global / BioASQ yes-no: the answer's opening Yes / No matches the gold label."""
    gold = (expectations or {}).get("expected_label")
    if gold not in ("yes", "no"):
        return None
    said = yes_no_label(answer_text(outputs))
    return Feedback(value=said == gold, rationale=f"answered {said}, gold {gold}")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


# Expression-system spellings -> the canonical names the BlueSky SMEs used in the gold.
_EXPRESSION_ALIASES = {
    "hek293t": "hek293", "hek293cells": "hek293",
    "cos1": "cos", "cos7": "cos", "coscells": "cos",
    "escherichiacoli": "ecoli", "bl21de3": "ecoli", "bl21": "ecoli",
    "sf21cells": "sf21", "sf9cells": "sf9",
    "nih3t3cells": "nih3t3",
}


def _expression_key(name: str) -> str:
    key = _norm(name)
    return _EXPRESSION_ALIASES.get(key, key)


def _answer_items(text: str) -> list[str]:
    arr = parse_js_array(text)
    if arr is not None:
        return [str(x) for x in arr]
    return [_strip(text).strip("'\"")]


@scorer
def expression_system_correct(inputs, outputs, expectations) -> Feedback | None:
    """BlueSky Q11: the extracted expression system matches one of the gold systems
    (normalized: case / punctuation, plus common aliases such as HEK293T -> HEK293)."""
    gold = (expectations or {}).get("expected_values")
    if kinase_question(inputs)[0] != "11" or not gold:
        return None
    text = answer_text(outputs)
    if is_sentinel(text):
        return Feedback(value=False, rationale=f"sentinel; gold {gold}")
    want = {_expression_key(g) for g in gold}
    got = {_expression_key(x) for x in _answer_items(text)}
    return Feedback(value=bool(want & got), rationale=f"extracted {sorted(got)}, gold {sorted(want)}")


_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
_FIG_NUM = re.compile(
    r"^(?P<supp>supplementa(?:ry|l)\s*|s(?=fig|table|t\d))?"
    r"(?P<kind>fig(?:ure)?s?|tables?|t)\.?\s*(?P<supp2>s)?(?P<num>\d+|[ivx]+\b)",
    re.IGNORECASE,
)


def figure_key(ref: str) -> str | None:
    """'Fig. 4B' -> 'fig4', 'Table III' -> 'table3', 'SFig.8' / 'Supplementary Figure S8' ->
    'sfig8', 'ST1' -> 'stable1'. Panel letters are dropped (the SMEs graded at figure level)."""
    m = _FIG_NUM.match(str(ref).strip())
    if not m:
        return None
    num = m.group("num").lower()
    num = str(_ROMAN[num]) if num in _ROMAN else num
    kind = "fig" if m.group("kind").lower().startswith("f") else "table"
    supp = "s" if (m.group("supp") or m.group("supp2")) else ""
    return f"{supp}{kind}{num}"


@scorer
def figure_ref_match(inputs, outputs, expectations) -> list | None:
    """BlueSky Q17: overlap between the answer's and the gold's figure / table refs.
    `figure_ref_match` = any overlap (how leniently the SMEs graded MyGPT);
    `figure_ref_recall` = share of the gold refs found."""
    gold = (expectations or {}).get("expected_values")
    if kinase_question(inputs)[0] != "17" or not gold:
        return None
    want = {k for k in (figure_key(g) for g in gold) if k}
    if not want:
        return None
    text = answer_text(outputs)
    got = set() if is_sentinel(text) else {k for k in (figure_key(x) for x in _answer_items(text)) if k}
    hit = want & got
    rationale = f"answer {sorted(got)}, gold {sorted(want)}"
    return [
        Feedback(name="figure_ref_match", value=bool(hit), rationale=rationale),
        Feedback(name="figure_ref_recall", value=len(hit) / len(want), rationale=rationale),
    ]


@scorer
def factoid_match(outputs, expectations) -> Feedback | None:
    """BioASQ factoid: the answer contains any gold entity synonym (normalized substring) —
    BioASQ's lenient accuracy."""
    gold = (expectations or {}).get("expected_values")
    if not gold or (expectations or {}).get("expected_label"):
        return None
    text = _norm(answer_text(outputs))
    hits = [g for g in gold if len(_norm(g)) > 1 and _norm(g) in text]
    return Feedback(value=bool(hits), rationale=f"matched {hits[:3]}" if hits else f"none of {gold[:5]}")


@scorer
def gold_correctness(inputs, outputs, expectations) -> Feedback | None:
    """MLflow's Correctness judge, only on rows that carry expected_response / expected_facts
    (the built-in raises on rows without them)."""
    from mlflow.genai.judges import is_correct

    exp = expectations or {}
    facts, response = exp.get("expected_facts"), exp.get("expected_response")
    if not facts and not response:
        return None
    return is_correct(
        request=_question(inputs), response=answer_text(outputs),
        expected_facts=facts or None, expected_response=None if facts else response,
        name="gold_correctness",
    )


# --------------------------------------------------------------------------- #
# scorer sets per use case
# --------------------------------------------------------------------------- #
_GROUNDING_GUIDELINES = Guidelines(
    name="mygpt_grounding",
    guidelines=[
        "The response only states facts that are supported by the documents the assistant retrieved; "
        "if the documents don't cover the question, it says so instead of guessing.",
        "The response answers in the language of the question (English), even when the source documents "
        "are in another language.",
    ],
)


def judges() -> list:
    return [
        RetrievalGroundedness(), RetrievalRelevance(), RelevanceToQuery(), Safety(),
        ExpectationsGuidelines(), _GROUNDING_GUIDELINES,
    ]


def scorers_for(use_case: str, vocab: list[str] | None = None) -> list:
    shared = [retrieval_ok, mygpt_trust, focus_adherence]
    if use_case == "sj_global":
        return judges() + shared + [yes_no_format, answer_label, yes_no_correct, gold_correctness]
    if use_case == "bluesky":
        extra = [make_vocab_compliance(vocab)] if vocab else []
        return judges() + shared + [
            js_array_valid, figure_ref_format, kinase_in_context,
            expression_system_correct, figure_ref_match, gold_correctness,
        ] + extra
    if use_case == "bioasq":
        # No focus_adherence (library-wide retrieval) and no yes_no_format (87 of the 163
        # are factoid questions); yes_no_correct / factoid_match skip the other type.
        return judges() + [retrieval_ok, mygpt_trust, yes_no_correct, factoid_match, gold_correctness]
    raise ValueError(f"Unknown use case {use_case!r}")
