"""Build MLflow GenAI eval records from St. Jude's MyGPT-evaluations gold data.

Pure functions (no MLflow / workspace calls) so they unit-test locally; the job
entrypoint ``build_datasets.py`` merges the records into UC-backed eval datasets.

Each record's ``inputs`` is the exact ``mygpt-rag-agent`` request body, so
``mlflow.genai.to_predict_fn("endpoints:/mygpt-rag-agent")`` passes it straight
through. Neither use case ships gold answers (Phase 2 = guidelines-only), so
``expectations`` carries per-row ``guidelines`` for the ExpectationsGuidelines judge;
``expected_facts`` / ``expected_response`` can be merged in later (SME labeling).

  * SJ Global (health_policies): every policy question is asked of EACH policy
    document — MyGPT's protocol passes ``document_title`` — so a row is
    (question x document), focused on that document.
  * BlueSky (kinase_literature): 3 templated extraction questions x 93
    (paper, kinase) pairs, focused on the paper, with the kinase synonyms appended
    and MyGPT's per-question answer-format prompt passed as ``instructions``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

EVAL_REPO = os.path.join("reference", "MyGPT-evaluations")
# The corpus PDFs as indexed (one folder per corpus); read via the /Volumes FUSE mount
# on Databricks. Locally (tests) the submodule's own PDF folders are used instead.
DOCS_VOLUME = "/Volumes/structural_biology/mygpt/docs"

# MyGPT-evaluations' `maximum_chunks_count` for both use cases.
K = 10


@dataclass(frozen=True)
class EvalDatasetSpec:
    uc_table: str  # UC table backing the MLflow eval dataset
    corpus: str  # custom_inputs.corpus — the top-level volume folder
    use_case: str


def specs(catalog: str = "structural_biology", schema: str = "mygpt") -> dict[str, EvalDatasetSpec]:
    return {
        "health_policies_fr": EvalDatasetSpec(f"{catalog}.{schema}.eval_health_policies_fr", "health_policies_fr", "sj_global"),
        "health_policies_es": EvalDatasetSpec(f"{catalog}.{schema}.eval_health_policies_es", "health_policies_es", "sj_global"),
        "kinase_literature": EvalDatasetSpec(f"{catalog}.{schema}.eval_kinase_literature", "kinase_literature", "bluesky"),
    }


def _read_csv(path: str) -> list[dict]:
    # The upstream CSVs mix UTF-8 (with BOM) and Latin-1; try in that order.
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def _request(question: str, corpus: str, document: str, instructions: str = "") -> dict:
    custom_inputs: dict = {"corpus": corpus, "focus": {"documents": [document]}, "k": K}
    if instructions:
        custom_inputs["instructions"] = instructions
    return {"input": [{"role": "user", "content": question}], "custom_inputs": custom_inputs}


# --------------------------------------------------------------------------- #
# SJ Global — health_policies
# --------------------------------------------------------------------------- #
_LANG = {"health_policies_fr": ("French", "French_10_pdfs"), "health_policies_es": ("Spanish", "Spanish_10_pdfs")}


def health_policy_records(base_dir: str, corpus: str, source_sha: str = "") -> list[dict]:
    language, pdf_dir = _LANG[corpus]
    root = os.path.join(base_dir, EVAL_REPO, "health_policies", "inputs")
    questions = _read_csv(os.path.join(root, "global_questions.csv"))
    # PDFs aren't bundle-synced (databricks.yml sync.exclude); list the indexed volume folder.
    volume_dir = os.path.join(DOCS_VOLUME, corpus)
    doc_dir = volume_dir if os.path.isdir(volume_dir) else os.path.join(root, pdf_dir)
    documents = sorted(f for f in os.listdir(doc_dir) if f.lower().endswith(".pdf"))
    records = []
    for doc in documents:
        for q in questions:
            records.append({
                "inputs": _request(q["question"], corpus, doc),
                "expectations": {"guidelines": [
                    "The response begins with 'Yes' or 'No'.",
                    "If the response begins with 'Yes', it follows with a short summary of what the "
                    "policy document says on the asked topic; if 'No', it does not describe policy "
                    "content on that topic.",
                    f"The response is written in English, even though the policy document is in {language}.",
                ]},
                "tags": {
                    "use_case": "sj_global", "corpus": corpus, "document": doc, "language": language.lower(),
                    "qid": q["qid"], "question_key": q["question_key"], "domain": q["domain"],
                    # for_testing=yes rows are the modified variants of Q1-5
                    "variant": q["for_testing"].lower(), "source_sha": source_sha,
                },
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
_KINASE_COMMON_GUIDELINE = (
    "If the retrieved paper does not mention the kinase (or one of its synonyms), the response is "
    "'no match' rather than an extracted value."
)


def kinase_synonyms(base_dir: str) -> dict[str, str]:
    path = os.path.join(base_dir, EVAL_REPO, "kinase_literature", "inputs", "human_kinome_synonyms.csv")
    return {r["kinase_ID"]: r["synonyms"] for r in _read_csv(path) if r.get("kinase_ID") and r.get("synonyms")}


def expression_systems(base_dir: str) -> list[str]:
    """Controlled vocabulary for Q11. Not in the upstream repo (collect_answers.py expects
    inputs/expression_systems_list.csv) — St. Jude to provide; empty until then."""
    path = os.path.join(base_dir, EVAL_REPO, "kinase_literature", "inputs", "expression_systems_list.csv")
    if not os.path.exists(path):
        return []
    return [r["expression_system_name"] for r in _read_csv(path) if r.get("expression_system_name")]


def kinase_records(base_dir: str, source_sha: str = "", vocab: list[str] | None = None) -> list[dict]:
    root = os.path.join(base_dir, EVAL_REPO, "kinase_literature", "inputs")
    questions = _read_csv(os.path.join(root, "questions.csv"))
    pairs = _read_csv(os.path.join(root, "kinase_pubmed_pairs.csv"))
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
            records.append({
                "inputs": _request(text, "kinase_literature", document,
                                   _KINASE_INSTRUCTIONS[qid].format(vocab=vocab_text)),
                "expectations": {"guidelines": _KINASE_GUIDELINES[qid] + [_KINASE_COMMON_GUIDELINE]},
                "tags": {
                    "use_case": "bluesky", "corpus": "kinase_literature", "document": document,
                    "pubmed_id": pmid, "doi": pair.get("doi", ""), "kinase": kinase, "question_id": qid,
                    "vocab_pending": str(qid == "11" and not vocab).lower(), "source_sha": source_sha,
                },
            })
    return records


def build_records(name: str, base_dir: str, source_sha: str = "") -> list[dict]:
    if name in _LANG:
        return health_policy_records(base_dir, name, source_sha)
    if name == "kinase_literature":
        return kinase_records(base_dir, source_sha)
    raise ValueError(f"Unknown eval dataset {name!r}")
