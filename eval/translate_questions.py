"""Translate the SJ Global base questions into French / Spanish once, for the Phase-6
native-language eval slice (eval_datasets.native_questions).

Uses the same `mygpt-translate` gateway (gemma-3-12b) and prompt as the RAG agent. Writes
eval/native_questions.csv (qid, language, question); review it, commit it, then re-run
mygpt_eval_build_datasets. Run locally with a CLI profile:

    python eval/translate_questions.py --profile mygpt [--languages french,spanish]
"""

from __future__ import annotations

import argparse
import csv
import os

LANGUAGES = {"french": ("French", "fr"), "spanish": ("Spanish", "es"), "hindi": ("Hindi", "hi")}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(ROOT, "reference", "MyGPT-evaluations", "health_policies", "inputs", "global_questions.csv")
OUT = os.path.join(ROOT, "eval", "native_questions.csv")


def system_prompt(target: str) -> str:
    """The agent's translator prompt (agents/rag_agent/agent.py _translate), English -> target."""
    full, code = LANGUAGES[target]
    return (
        f"You are a professional English (en) to {full} ({code}) translator. Your goal is to accurately "
        f"convey the meaning and nuances of the original English text while adhering to {full} grammar, "
        f"vocabulary, and cultural sensitivities. Produce only the {full} translation, without any "
        f"additional explanations or commentary. If the text is already in {full}, return it unchanged. "
        f"Please translate the following English text into {full}: "
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="mygpt")
    ap.add_argument("--endpoint", default="mygpt-translate")
    ap.add_argument("--languages", default="french,spanish")
    args = ap.parse_args()

    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient(profile=args.profile)
    with open(QUESTIONS, newline="", encoding="utf-8-sig") as f:
        base = [r for r in csv.DictReader(f) if r["for_testing"].strip().lower() == "no"]
    rows = []
    for language in [x.strip() for x in args.languages.split(",") if x.strip()]:
        for r in base:
            resp = w.api_client.do("POST", f"/serving-endpoints/{args.endpoint}/invocations", body={
                "messages": [{"role": "system", "content": system_prompt(language)},
                             {"role": "user", "content": r["question"].strip()}],
                "temperature": 0})
            text = str(resp["choices"][0]["message"]["content"]).strip()
            rows.append({"qid": r["qid"], "language": language, "question": text})
            print(f"[{language}] q{r['qid']}: {text[:80]}")
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["qid", "language", "question"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} questions to {OUT}")


if __name__ == "__main__":
    main()
