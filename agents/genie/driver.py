"""Create or update the MyGPT Genie space over the structured seed tables (Phase 5).

Run as a DABs job task (`mygpt_deploy_genie`) after `mygpt_etl` has built the tables
(etl/structured_seed.py). DABs has no Genie resource, so this follows the driver-script
pattern: the space is found by TITLE and updated in place, otherwise created. The Supervisor
driver resolves the space id by the same title, so there's no id to copy around.

The space definition is the Genie `serialized_space` (version 2) — tables, text
instructions, sample questions and example SQL. Ids are derived from the content so
re-running the driver doesn't churn them.

    python agents/genie/driver.py --catalog structural_biology --schema mygpt --warehouse-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json

TITLE = "MyGPT structured data"

DESCRIPTION = (
    "Structured data behind MyGPT: the human kinome (kinase groups, families, UniProt ids, "
    "synonyms), the BlueSky kinase literature library (which papers study which kinase) and the "
    "catalogue of governed documents per corpus. Answers counts, lists and joins; it does NOT "
    "read document text (ask MyGPT's document search for what a paper says)."
)

INSTRUCTIONS = [
    "Kinases are identified by gene symbol in kinome.kinase_id (e.g. ABL1). When a user names a "
    "kinase by protein name or alias, match it against kinome.synonyms with "
    "array_contains or exists(synonyms, s -> lower(s) = lower('<name>')).\n",
    "Join kinase_papers to kinome on kinase_id, and kinase_papers to corpus_documents on source_name.\n",
    "A paper is 'in the library' when it has a row in corpus_documents with n_chunks > 0.\n",
    "kinome.kinase_group is the Manning group (AGC, CAMK, CK1, CMGC, RGC, STE, TK, TKL, Atypical, Other).\n",
    "Corpora: biomed = BioASQ biomedical papers; kinase_literature = BlueSky kinase papers; "
    "health_policies_fr / health_policies_es = SJ Global health policies in French / Spanish.\n",
]

SAMPLE_QUESTIONS = [
    "How many kinases are in each kinase group?",
    "Which kinases in the CMGC group have papers in the library?",
    "List the papers about MAPK14 with their DOIs",
    "How many documents does each corpus contain?",
]


def example_sqls(fq: str) -> list[tuple[str, str]]:
    return [
        ("Which kinases in the CMGC group have papers in the library?",
         f"SELECT k.kinase_id, k.kinase_family, count(DISTINCT p.pubmed_id) AS papers\n"
         f"FROM {fq}.kinome k\n"
         f"JOIN {fq}.kinase_papers p ON p.kinase_id = k.kinase_id\n"
         f"JOIN {fq}.corpus_documents d ON d.source_name = p.source_name AND d.n_chunks > 0\n"
         f"WHERE k.kinase_group = 'CMGC'\n"
         f"GROUP BY k.kinase_id, k.kinase_family\n"
         f"ORDER BY papers DESC, k.kinase_id"),
        ("How many documents does each corpus contain?",
         f"SELECT corpus, count(*) AS documents, sum(CASE WHEN n_chunks > 0 THEN 1 ELSE 0 END) AS indexed\n"
         f"FROM {fq}.corpus_documents\n"
         f"GROUP BY corpus\n"
         f"ORDER BY corpus"),
    ]


def _id(*parts: str) -> str:
    return hashlib.md5("\x1f".join(parts).encode()).hexdigest()


def _lines(text: str) -> list[str]:
    """Genie stores multi-line text as a list of newline-terminated lines."""
    out = text.splitlines(keepends=True)
    return out or [""]


def serialized_space(fq: str) -> dict:
    tables = sorted(["kinome", "kinase_papers", "corpus_documents"])
    return {
        "version": 2,
        "config": {
            "sample_questions": sorted(
                [{"id": _id("q", q), "question": [q]} for q in SAMPLE_QUESTIONS], key=lambda x: x["id"]
            ),
        },
        "data_sources": {
            # Descriptions come from the table / column COMMENTs written by structured_seed.py.
            "tables": [{"identifier": f"{fq}.{t}"} for t in tables],
        },
        "instructions": {
            "text_instructions": [{"id": _id("instructions"), "content": INSTRUCTIONS}],
            "example_question_sqls": sorted(
                [{"id": _id("sql", q), "question": [q], "sql": _lines(sql)} for q, sql in example_sqls(fq)],
                key=lambda x: x["id"],
            ),
        },
    }


def find_space(w, title: str) -> str | None:
    token = None
    while True:
        page = w.api_client.do("GET", "/api/2.0/genie/spaces", query={"page_token": token} if token else None)
        for s in page.get("spaces") or []:
            if s.get("title") == title:
                return s["space_id"]
        token = page.get("next_page_token")
        if not token:
            return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--warehouse-id", required=True)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--base-dir", default="")  # accepted for job parity; unused here
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    fq = f"{args.catalog}.{args.schema}"
    body = {
        "title": args.title,
        "description": DESCRIPTION,
        "warehouse_id": args.warehouse_id,
        "serialized_space": json.dumps(serialized_space(fq)),
    }
    space_id = find_space(w, args.title)
    if space_id:
        w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
        print(f"Updated Genie space {args.title!r} ({space_id})")
    else:
        space_id = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)["space_id"]
        print(f"Created Genie space {args.title!r} ({space_id})")
    print(f"Managed MCP: {w.config.host}/api/2.0/mcp/genie/{space_id}")


if __name__ == "__main__":
    main()
