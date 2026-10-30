"""Build MLflow GenAI eval records from St. Jude's MyGPT-evaluations gold data.

Pure functions (no MLflow / workspace calls) so they unit-test locally; the job
entrypoint ``build_datasets.py`` merges the records into UC-backed eval datasets.

Each record's ``inputs`` is the exact ``mygpt-rag-agent`` request body, so
``mlflow.genai.to_predict_fn("endpoints:/mygpt-rag-agent")`` passes it straight
through. ``expectations`` always carries per-row ``guidelines`` for the
ExpectationsGuidelines judge. Where gold exists it also carries ``expected_label``
(yes / no), ``expected_values`` (accepted extraction values), ``expected_response`` or
``expected_facts`` (for the correctness judge); rows without gold carry none of these,
and the gold scorers skip them.

Gold comes from St. Jude's PRIVATE ground truths (``private_dir()``: a restricted UC
volume on Databricks, the gitignored ``MyGPT-evaluations-private-main`` locally) and,
for BioASQ, from the public benchmark file. Without the private folder the SJ Global /
BlueSky records are built exactly as before (guidelines only).

  * SJ Global (health_policies): every policy question is asked of EACH policy
    document — MyGPT's protocol passes ``document_title`` — so a row is
    (question x document), focused on that document. Phase 6 adds:
      - ``health_policies_hi``, the public Hindi corpus (guidelines only, no SME gold);
      - a native-language slice on FR / ES: the base questions asked in the documents'
        language (``eval/native_questions.csv``, made once by translate_questions.py),
        tagged ``gold = native`` and inheriting the base question's yes/no label.
  * BlueSky (kinase_literature): 3 templated extraction questions x 94
    (paper, kinase) pairs, focused on the paper, with the kinase synonyms appended
    and MyGPT's per-question answer-format prompt passed as ``instructions``.
  * BioASQ (public benchmark): 163 questions (76 yes/no + 87 factoid) asked of the
    whole ``biomed`` corpus — MyGPT's protocol has no document focus — with k = 15.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import sys
from dataclasses import dataclass

EVAL_REPO = os.path.join("reference", "MyGPT-evaluations")
# St. Jude's private ground truths: never in git or the bundle sync. On Databricks they
# live in a restricted UC volume (`make eval-private`); locally in the gitignored folder.
PRIVATE_VOLUME = "/Volumes/structural_biology/mygpt/eval_private"
PRIVATE_REPO = os.path.join("reference", "MyGPT-evaluations-private-main", "ground truths")
# The corpus PDFs as indexed (one folder per corpus); read via the /Volumes FUSE mount
# on Databricks. Locally (tests) the submodule's own PDF folders are used instead.
DOCS_VOLUME = "/Volumes/structural_biology/mygpt/docs"
# MyGPT-evaluations' CSVs at their repo paths under <volume>/MyGPT-evaluations (deploy.sh
# step 6). Without it, the bundle-synced reference/MyGPT-evaluations of a dev checkout.
REFERENCE_VOLUME = "/Volumes/structural_biology/mygpt/reference"


def configure(docs_volume: str = "", private_volume: str = "", reference_volume: str = "") -> None:
    """Point at the deployment's volumes (the jobs pass the bundle's catalog/schema/volume names)."""
    global DOCS_VOLUME, PRIVATE_VOLUME, REFERENCE_VOLUME
    DOCS_VOLUME = docs_volume or DOCS_VOLUME
    PRIVATE_VOLUME = private_volume or PRIVATE_VOLUME
    REFERENCE_VOLUME = reference_volume or REFERENCE_VOLUME


def eval_repo(base_dir: str) -> str:
    """Root of the MyGPT-evaluations inputs: the reference volume, else the synced repo copy."""
    volume_repo = os.path.join(REFERENCE_VOLUME, "MyGPT-evaluations")
    return volume_repo if os.path.isdir(volume_repo) else os.path.join(base_dir, EVAL_REPO)

# MyGPT-evaluations' `maximum_chunks_count` for both use cases (BioASQ uses 15).
K = 10
K_BIOASQ = 15


@dataclass(frozen=True)
class EvalDatasetSpec:
    uc_table: str  # UC table backing the MLflow eval dataset
    corpus: str  # custom_inputs.corpus — the top-level volume folder
    use_case: str


def specs(catalog: str = "structural_biology", schema: str = "mygpt") -> dict[str, EvalDatasetSpec]:
    return {
        "health_policies_fr": EvalDatasetSpec(f"{catalog}.{schema}.eval_health_policies_fr", "health_policies_fr", "sj_global"),
        "health_policies_es": EvalDatasetSpec(f"{catalog}.{schema}.eval_health_policies_es", "health_policies_es", "sj_global"),
        "health_policies_hi": EvalDatasetSpec(f"{catalog}.{schema}.eval_health_policies_hi", "health_policies_hi", "sj_global"),
        "kinase_literature": EvalDatasetSpec(f"{catalog}.{schema}.eval_kinase_literature", "kinase_literature", "bluesky"),
        "bioasq": EvalDatasetSpec(f"{catalog}.{schema}.eval_bioasq", "biomed", "bioasq"),
    }


def private_dir(base_dir: str) -> str | None:
    """Folder holding St. Jude's private ground-truth CSVs, or None (= no SME gold)."""
    for path in (PRIVATE_VOLUME, os.path.join(base_dir, PRIVATE_REPO)):
        if os.path.isdir(path):
            return path
    return None


def _read_csv(path: str) -> list[dict]:
    # The upstream CSVs mix UTF-8 (with BOM), Windows-1252 and Latin-1; try in that order.
    # BioASQ's concatenated gold abstracts exceed the csv module's default field limit.
    csv.field_size_limit(sys.maxsize)
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def _request(question: str, corpus: str, document: str | None, instructions: str = "", k: int = K,
             query_language: str | None = None) -> dict:
    custom_inputs: dict = {"corpus": corpus, "k": k}
    if query_language:
        custom_inputs["query_language"] = query_language
    if document:
        custom_inputs["focus"] = {"documents": [document]}
    if instructions:
        custom_inputs["instructions"] = instructions
    return {"input": [{"role": "user", "content": question}], "custom_inputs": custom_inputs}


# --------------------------------------------------------------------------- #
# SJ Global — health_policies
# --------------------------------------------------------------------------- #
# corpus -> (language, submodule PDF folder used when the docs volume isn't mounted).
# The Hindi corpus is public PDFs St. Jude didn't evaluate: volume only, no SME gold.
_LANG = {
    "health_policies_fr": ("French", "French_10_pdfs"),
    "health_policies_es": ("Spanish", "Spanish_10_pdfs"),
    "health_policies_hi": ("Hindi", None),
}
POLICY_GOLD = "Ground truths Global policies.csv"
# Phase 6: the base questions translated once into the documents' language (qid, language,
# question); written by eval/translate_questions.py and committed, so builds are deterministic.
NATIVE_QUESTIONS = os.path.join("eval", "native_questions.csv")


def native_questions(base_dir: str) -> dict[tuple[str, str], str]:
    """(qid, language) -> the question in that language; empty until the file exists."""
    path = os.path.join(base_dir, NATIVE_QUESTIONS)
    if not os.path.exists(path):
        return {}
    return {(r["qid"], r["language"].lower()): r["question"] for r in _read_csv(path) if r.get("question")}


def yes_no_label(text: str) -> str:
    """yes / no / unparseable from an answer's opening word (EN, FR, ES)."""
    m = re.match(r"^[\s*_#>\"']*(yes|no|oui|non|s[ií])\b", text.strip(), flags=re.IGNORECASE)
    if not m:
        return "unparseable"
    word = m.group(1).lower()
    return "yes" if word in ("yes", "oui", "si", "sí") else "no"


def policy_gold(base_dir: str) -> dict[tuple[str, str], dict]:
    """(document stem, question_key) -> gold, from the SME-labelled MyGPT answers.

    Each row is MyGPT's `answer_1` plus an SME Correct / Incorrect label. The questions
    are yes/no, so the gold label is the answer's own yes/no when Correct and the flip
    when Incorrect; the answer text is a reference summary only when Correct.
    Unlabelled rows (or answers with no yes/no opening) yield no gold."""
    folder = private_dir(base_dir)
    if folder is None:
        return {}
    gold = {}
    for r in _read_csv(os.path.join(folder, POLICY_GOLD)):
        label = r.get("Correctness for Answer 1 (Correct/Incorrect)", "").lower()
        said = yes_no_label(r.get("answer_1", ""))
        if label not in ("correct", "incorrect") or said == "unparseable":
            continue
        correct = label == "correct"
        gold[(r["document (Global policy)"], r["question_keyword"])] = {
            "expected_label": said if correct else ("no" if said == "yes" else "yes"),
            "expected_response": r["answer_1"] if correct else None,
            "mygpt_label": "correct" if correct else "incorrect",
        }
    return gold


def health_policy_records(base_dir: str, corpus: str, source_sha: str = "") -> list[dict]:
    language, pdf_dir = _LANG[corpus]
    root = os.path.join(eval_repo(base_dir), "health_policies", "inputs")
    questions = _read_csv(os.path.join(root, "global_questions.csv"))
    # PDFs aren't bundle-synced (databricks.yml sync.exclude); list the indexed volume folder.
    volume_dir = os.path.join(DOCS_VOLUME, corpus)
    doc_dir = volume_dir if os.path.isdir(volume_dir) else (os.path.join(root, pdf_dir) if pdf_dir else None)
    if doc_dir is None or not os.path.isdir(doc_dir):
        return []
    documents = sorted(f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf"))
    gold = policy_gold(base_dir)
    native = native_questions(base_dir)
    records = []
    for doc in documents:
        for q in questions:
            variant = q["for_testing"].lower()
            expectations: dict = {"guidelines": [
                "The response begins with 'Yes' or 'No'.",
                "If the response begins with 'Yes', it follows with a short summary of what the "
                "policy document says on the asked topic; if 'No', it does not describe policy "
                "content on that topic.",
                f"The response is written in English, even though the policy document is in {language}.",
            ]}
            # The SMEs labelled the base questions; a variant (Q6-10) is the same question
            # minus the keyword hints, so it inherits the base question's yes/no gold, but
            # NOT its reference summary (written for the hinted question, it penalizes correct
            # variant answers). `gold_scope` makes gold_correctness skip variants even if an
            # older build's expected_response survives a merge_records upsert.
            g = gold.get((doc.removesuffix(".pdf"), q["question_key"]))
            if g:
                expectations["expected_label"] = g["expected_label"]
                if variant == "yes":
                    expectations["gold_scope"] = "label_only"
                elif g["expected_response"]:
                    expectations["expected_response"] = g["expected_response"]
            records.append({
                "inputs": _request(q["question"], corpus, doc),
                "expectations": expectations,
                "tags": {
                    "use_case": "sj_global", "corpus": corpus, "document": doc, "language": language.lower(),
                    "qid": q["qid"], "question_key": q["question_key"], "domain": q["domain"],
                    # for_testing=yes rows are the modified variants of Q1-5
                    "variant": variant, "source_sha": source_sha,
                    "gold": ("inherited" if variant == "yes" else "sme") if g else "none",
                    "mygpt_label": g["mygpt_label"] if g else "",
                },
            })
            # Phase 6: the base question asked in the documents' language. It inherits only
            # the yes/no label (the reference summary is English, for the hinted wording).
            native_q = native.get((q["qid"], language.lower())) if variant == "no" else None
            if native_q:
                native_exp: dict = {"guidelines": [
                    "The response begins with 'Yes' or 'No' (or the same word in the language of the question).",
                    expectations["guidelines"][1],
                    f"The response is written in {language}, the language of the question.",
                ]}
                if g:
                    native_exp["expected_label"] = g["expected_label"]
                    native_exp["gold_scope"] = "label_only"
                records.append({
                    "inputs": _request(native_q, corpus, doc, query_language=language.lower()),
                    "expectations": native_exp,
                    "tags": {**records[-1]["tags"], "query_language": language.lower(),
                             "gold": "native" if g else "none"},
                })
    return records


# --------------------------------------------------------------------------- #
# BlueSky — kinase_literature
# --------------------------------------------------------------------------- #
# MyGPT's per-question answer-format prompts, ported from
# kinase_literature/scripts/collect_answers.py:129-137 (minus the inlined contexts — the
# agent supplies its own retrieved sources). That script's hard-coded question ids
# (11/15/16) are stale vs. questions.csv (11/16v2/17); the prompts map by row position:
# expression system -> assay methods -> figures/tables.
_KINASE_INSTRUCTIONS = {
    "11": (
        "Answer in less than 50 words, with minimal words. Answer the expression system and avoid "
        "other text. If there is no expression system in these experiments, answer with \"no answer\". "
        "{vocab}Format it as a JavaScript array with a single value of expression system, no other text."
    ),
    "16v2": (
        "Answer in less than 50 words, with minimal text. If no information is available, respond with "
        "\"no match\". If information is available, list the assay methods used to detect substrate "
        "phosphorylation, kinase activity, or catalytic activity, ensuring that the mentioned kinase was "
        "the enzyme in the experiment. Only list the method names as a JavaScript array, with no additional text."
    ),
    "17": (
        "Answer in less than 50 words, with minimal words. List the figures or tables and avoid other "
        "text. If there are no figures or tables in these experiments, answer with \"no answer\". Format "
        "it as a JavaScript array of figures and tables, no other text."
    ),
}
_KINASE_GUIDELINES = {
    "11": ["The response is only a JavaScript array with a single expression-system name, or 'no match' / 'no answer'."],
    "16v2": ["The response is only a JavaScript array of assay method names, or 'no match' / 'no answer'."],
    "17": ["The response is only a JavaScript array of figure / table references, or 'no match' / 'no answer'."],
}
# Not a per-row guideline: "if the paper doesn't mention the kinase, answer 'no match'" can't
# be judged by ExpectationsGuidelines, which sees only the question + answer, not the paper
# (it failed 176/282 rows on that clause alone in the first baseline). The code scorer
# `no_match_consistent` checks it against the retrieved chunks instead.


def kinase_synonyms(base_dir: str) -> dict[str, str]:
    path = os.path.join(eval_repo(base_dir), "kinase_literature", "inputs", "human_kinome_synonyms.csv")
    return {r["kinase_ID"]: r["synonyms"] for r in _read_csv(path) if r.get("kinase_ID") and r.get("synonyms")}


def expression_systems(base_dir: str) -> list[str]:
    """Controlled vocabulary for Q11. Not in the upstream repo (collect_answers.py expects
    inputs/expression_systems_list.csv) — St. Jude to provide; empty until then."""
    path = os.path.join(eval_repo(base_dir), "kinase_literature", "inputs", "expression_systems_list.csv")
    if not os.path.exists(path):
        return []
    return [r["expression_system_name"] for r in _read_csv(path) if r.get("expression_system_name")]


KINASE_GOLD = "Ground truths Kinases Literature.csv"
# The ground-truth file keeps collect_answers.py's question ids; they map by position onto
# questions.csv (expression system -> assay methods -> figures/tables), as above.
_GOLD_QID = {"11": "11", "15": "16v2", "16": "17"}
# MyGPT's three SME-graded answer configs in the ground-truth file: (answer, label) columns.
KINASE_MYGPT_CONFIGS = {
    "ctx5": ("answer_context_5", "label_context_5"),
    "ctx15": ("answer_context_15", "label_context_15"),
    "strpre": ("answer_str_pre", "label_str_pre"),
}


def _split_gold(value: str) -> list[str]:
    seen: list[str] = []
    for part in value.split(";"):
        part = part.strip().rstrip(".").strip()
        if part and part not in seen:
            seen.append(part)
    return seen


_FIG_LIKE = re.compile(r"^(s?fig|s?table|st\d|table)", re.IGNORECASE)


def _text_qid(question: str) -> str | None:
    """Our question id from a ground-truth row's question TEXT (see scorers.kinase_question)."""
    if "what is the expression system" in question:
        return "11"
    if "what assay was used" in question:
        return "16v2"
    if "which figure or table" in question:
        return "17"
    return None


def _gold_fits(qid: str, ground_truth: str) -> bool:
    """Shape check: figure questions have figure / table refs as gold, the others don't."""
    return (qid == "17") == bool(_FIG_LIKE.match(ground_truth.strip()))


def kinase_gold_rows(base_dir: str) -> dict[tuple[str, str, str], dict]:
    """(pubmed_id, kinase, our question_id) -> the ground-truth row (all 282, including the
    ones without a `ground_truth`, which still carry MyGPT's SME labels).

    Rows are keyed by their question TEXT, which agrees with the pubmed / kinase / answer /
    label columns. In one block of St. Jude's file (ids 129-262) the `question_id` and
    `ground_truth` cells sit one row too high, so for a row whose `question_id` disagrees with
    its text, the gold is the previous row's (when that row is in the shifted block too).
    Such rows are tagged `realigned`; any gold that still doesn't fit its question is dropped.
    """
    folder = private_dir(base_dir)
    if folder is None:
        return {}
    rows = _read_csv(os.path.join(folder, KINASE_GOLD))
    out = {}
    prev_shifted, prev_gt = False, ""
    for r in rows:
        qid = _text_qid(r.get("Question", ""))
        if qid is None:
            continue
        shifted = _GOLD_QID.get(r.get("question_id", "")) != qid
        gt = (prev_gt if prev_shifted else "") if shifted else r.get("ground_truth", "")
        if gt and not _gold_fits(qid, gt):
            gt = ""
        out[(r["pubmed_id"], r["kinase"], qid)] = {**r, "ground_truth": gt, "realigned": shifted}
        prev_shifted, prev_gt = shifted, r.get("ground_truth", "")
    return out


def kinase_expectations(qid: str, ground_truth: str) -> dict:
    """Gold expectations for one kinase row; {} when the SMEs gave no ground truth."""
    values = _split_gold(ground_truth)
    if not values:
        return {}
    if qid == "16v2":
        # The gold is a coarse assay category ("protein kinase assay") while answers name
        # specific methods, so it's graded by the correctness judge, not string match.
        return {"expected_facts": [f"The assay method named is a type of {v}." for v in values]}
    return {"expected_values": values}


def kinase_records(base_dir: str, source_sha: str = "", vocab: list[str] | None = None) -> list[dict]:
    root = os.path.join(eval_repo(base_dir), "kinase_literature", "inputs")
    questions = _read_csv(os.path.join(root, "questions.csv"))
    pairs = _read_csv(os.path.join(root, "kinase_pubmed_pairs.csv"))
    gold = kinase_gold_rows(base_dir)
    synonyms = kinase_synonyms(base_dir)
    vocab = expression_systems(base_dir) if vocab is None else vocab
    vocab_text = f"Use one of these choices for expression system: [{', '.join(vocab)}]. " if vocab else ""
    records = []
    for pair in pairs:
        kinase, pmid = pair["kinase_name"], pair["pubmed_id"]
        # PDFs are named by PubMed id (MyGPT contexts read "Page N - <pmid>: ..."); a
        # `pdf_name` column, if St. Jude's delivery adds one, overrides that.
        document = pair.get("pdf_name") or f"{pmid}.pdf"
        for q in questions:
            qid = q["question_id"]
            text = q["question"].replace("[kinase name]", kinase)
            if kinase in synonyms:
                text += f" The kinase {kinase} is also known as {synonyms[kinase]}."
            g = gold.get((pmid, kinase, qid)) or {}
            expected = kinase_expectations(qid, g.get("ground_truth", ""))
            records.append({
                "inputs": _request(text, "kinase_literature", document,
                                   _KINASE_INSTRUCTIONS[qid].format(vocab=vocab_text)),
                "expectations": {"guidelines": _KINASE_GUIDELINES[qid], **expected},
                "tags": {
                    "use_case": "bluesky", "corpus": "kinase_literature", "document": document,
                    "pubmed_id": pmid, "doi": pair.get("doi", ""), "kinase": kinase, "question_id": qid,
                    "vocab_pending": str(qid == "11" and not vocab).lower(), "source_sha": source_sha,
                    "gold": ("sme_realigned" if g.get("realigned") else "sme") if expected else "none",
                    # MyGPT's SME-graded correctness per config (parity / judge-agreement reports)
                    **{f"mygpt_label_{c}": g.get(lab, "").lower() for c, (_a, lab) in KINASE_MYGPT_CONFIGS.items()},
                },
            })
    return records


# --------------------------------------------------------------------------- #
# BioASQ — public biomedical QA benchmark (13b golden enriched subset)
# --------------------------------------------------------------------------- #
BIOASQ_DIR = os.path.join("BioASQ", "inputs")  # under eval_repo()
# MyGPT's answer prompt (BioASQ/scripts/collect_answers.py:101, minus the inlined contexts).
# It forces a Yes / No / Maybe opening on every question; we keep that for yes/no questions
# only, since it makes no sense for a factoid.
_BIOASQ_INSTRUCTIONS = {
    "yesno": "Answer the question in less than 100 words, and with minimal amount of words, try not to use "
             "anything else. Always start the answer with \"Yes,\", \"No,\" or \"Maybe,\" and then answer the question.",
    "factoid": "Answer the question in less than 100 words, and with minimal amount of words, try not to use "
               "anything else.",
}
_BIOASQ_GUIDELINES = {
    "yesno": ["The response begins with 'Yes', 'No' or 'Maybe' and then briefly answers the question."],
    "factoid": ["The response directly names the asked-for entity or value in under 100 words."],
}


def _literal(value: str):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _flatten(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _flatten(v)]
    return [str(value).strip()] if str(value).strip() else []


def bioasq_records(base_dir: str, source_sha: str = "") -> list[dict]:
    root = os.path.join(eval_repo(base_dir), BIOASQ_DIR)
    # questions_final.csv's `pubmed_id` column actually holds the BioASQ question id.
    questions = _read_csv(os.path.join(root, "questions_final.csv"))
    gold = {r["question_id"]: r for r in _read_csv(os.path.join(root, "bioasq_13b_final_3964.csv"))}
    records = []
    for q in questions:
        qid = q["pubmed_id"]
        g = gold[qid]
        qtype = g["type"]
        ideal = _flatten(_literal(g["ideal_answer"]))
        expectations: dict = {"guidelines": _BIOASQ_GUIDELINES[qtype]}
        if qtype == "yesno":
            expectations["expected_label"] = g["exact_answer"].strip().lower()
        else:
            # exact_answer = [[synonym, ...], ...] — any synonym of any entity counts
            expectations["expected_values"] = _flatten(_literal(g["exact_answer"]))
        if ideal:
            expectations["expected_response"] = ideal[0]
        records.append({
            "inputs": _request(q["questions"], "biomed", None, _BIOASQ_INSTRUCTIONS[qtype], k=K_BIOASQ),
            "expectations": expectations,
            "tags": {"use_case": "bioasq", "corpus": "biomed", "bioasq_id": qid, "question_type": qtype,
                     "gold": "benchmark", "source_sha": source_sha},
        })
    return records


def build_records(name: str, base_dir: str, source_sha: str = "") -> list[dict]:
    if name in _LANG:
        return health_policy_records(base_dir, name, source_sha)
    if name == "kinase_literature":
        return kinase_records(base_dir, source_sha)
    if name == "bioasq":
        return bioasq_records(base_dir, source_sha)
    raise ValueError(f"Unknown eval dataset {name!r}")
