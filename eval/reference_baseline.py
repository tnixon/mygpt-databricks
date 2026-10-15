"""MyGPT parity baseline + judge agreement, from MyGPT's own recorded answers.

DABs job: mygpt_eval_reference. No agent endpoint calls: MyGPT's published answers (in
MyGPT-evaluations and St. Jude's private ground truths) are scored with OUR gold scorers
(eval/scorers.py), and the results are logged to one MLflow run in the shared experiment:

  * mygpt_ref/<use_case>/<config>/sme_accuracy   St. Jude's SME grading of MyGPT — the number
                                                  our agent is compared against;
  * mygpt_ref/<use_case>/<config>/<scorer>        our scorers' accuracy on the same answers;
  * mygpt_ref/<use_case>/<config>/<scorer>/agreement
                                                  how often that scorer agrees with the SMEs
                                                  (where both exist) — i.e. how far to trust it
                                                  as a gate;
  * reference_rows.json                           every scored row, for the disagreements.

What each source supports:
  * SJ Global: SME accuracy only. Our policy gold is DERIVED from these very answers + their
    SME labels (eval_datasets.policy_gold), so scoring them against it would agree by
    construction.
  * BlueSky: three answer configs (context_5 / context_15 / str_pre), each SME-graded, plus an
    independent ground truth -> accuracy and agreement for expression_system_correct (Q11),
    figure_ref_match (Q17) and gold_correctness (Q16v2).
  * BioASQ: MyGPT's gpt-oss-20b answers vs the benchmark gold -> accuracy (no SME labels).

Usage:
    python eval/reference_baseline.py [--judge true] [--experiment /Shared/mygpt/mygpt-rag-agent]
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

BIOASQ_ANSWERS = os.path.join("reference", "MyGPT-evaluations", "BioASQ", "outputs", "answers",
                              "answers-gpt-oss-20b-BioASQ.csv")


def _base_dir(base_dir: str) -> str:
    if base_dir:
        return base_dir
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.getcwd()


def _sme(label: str) -> bool | None:
    label = (label or "").strip().lower()
    return {"correct": True, "incorrect": False}.get(label)


def reference_rows(base: str) -> list[dict]:
    """One row per (use case, MyGPT config, dataset record) with MyGPT's answer as `outputs`."""
    import eval_datasets as ed

    rows: list[dict] = []
    # SJ Global: the SME label is the whole story (see the module docstring).
    for (doc, key), g in ed.policy_gold(base).items():
        rows.append({"use_case": "sj_global", "config": "answer_1", "key": f"{doc}|{key}",
                     "inputs": None, "outputs": None, "expectations": {}, "sme": g["mygpt_label"] == "correct"})

    # BlueSky: every SME-graded answer config, scored against the independent ground truth.
    gold_rows = ed.kinase_gold_rows(base)
    for rec in ed.kinase_records(base):
        t = rec["tags"]
        g = gold_rows.get((t["pubmed_id"], t["kinase"], t["question_id"]))
        if not g:
            continue
        for config, (answer_col, label_col) in ed.KINASE_MYGPT_CONFIGS.items():
            rows.append({
                "use_case": "bluesky", "config": config, "question_id": t["question_id"], "gold": t["gold"],
                "key": f"{t['pubmed_id']}|{t['kinase']}|{t['question_id']}",
                "inputs": rec["inputs"], "outputs": g.get(answer_col, ""),
                "expectations": rec["expectations"], "sme": _sme(g.get(label_col, "")),
            })

    # BioASQ: MyGPT's gpt-oss-20b answers vs the benchmark gold.
    path = os.path.join(base, BIOASQ_ANSWERS)
    if os.path.exists(path):
        answers = {r["pubmed_id"]: r["answer"] for r in ed._read_csv(path)}
        for rec in ed.bioasq_records(base):
            t = rec["tags"]
            if t["bioasq_id"] in answers:
                rows.append({"use_case": "bioasq", "config": "gpt-oss-20b", "question_type": t["question_type"],
                             "key": t["bioasq_id"], "inputs": rec["inputs"], "outputs": answers[t["bioasq_id"]],
                             "expectations": rec["expectations"], "sme": None})
    return rows


def _feedbacks(result, default_name: str) -> dict:
    """Scorer result (Feedback / list of Feedback / bool / None) -> {name: value}. Called
    outside evaluate(), an unnamed Feedback carries MLflow's placeholder name, so single
    results take the scorer's name; multi-feedback scorers name each one themselves."""
    if result is None:
        return {}
    if not isinstance(result, list):
        return {default_name: getattr(result, "value", result)}
    return {fb.name: fb.value for fb in result}


def score_row(row: dict, judge: bool) -> dict:
    import scorers as sc

    if row["inputs"] is None:
        return {}
    kwargs = {"inputs": row["inputs"], "outputs": row["outputs"], "expectations": row["expectations"]}
    code = {
        "bluesky": [("expression_system_correct", sc.expression_system_correct), ("figure_ref_match", sc.figure_ref_match)],
        "bioasq": [("yes_no_correct", sc.yes_no_correct), ("factoid_match", sc.factoid_match)],
    }[row["use_case"]]
    values: dict = {}
    for name, fn in code:
        fb = _feedbacks(_call(fn, **kwargs), name)
        values.update({k: v for k, v in fb.items() if k != "figure_ref_recall"})
    if judge:
        values.update(_feedbacks(_call(sc.gold_correctness, **kwargs), "gold_correctness"))
    return values


def _call(fn, **kwargs):
    """Call a scorer with only the arguments its signature takes."""
    import inspect

    params = inspect.signature(getattr(fn, "_original_func", None) or fn).parameters
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "yes", "1", "1.0")


def summarize(rows: list[dict], scored: list[dict]) -> dict[str, float]:
    """Per (use case, config), and for BlueSky also per gold provenance (`sme` vs
    `sme_realigned`), so a wrong realignment of St. Jude's shifted block would show up as a
    lower agreement on that slice."""
    groups: dict[tuple[str, ...], list[tuple[dict, dict]]] = {}
    for row, values in zip(rows, scored):
        groups.setdefault((row["use_case"], row["config"]), []).append((row, values))
        if row.get("gold") in ("sme", "sme_realigned"):
            groups.setdefault((row["use_case"], row["config"], row["gold"]), []).append((row, values))
    metrics: dict[str, float] = {}
    for group, items in groups.items():
        prefix = "mygpt_ref/" + "/".join(group)
        sme = [r["sme"] for r, _ in items if r["sme"] is not None]
        if sme:
            metrics[f"{prefix}/sme_accuracy"] = sum(sme) / len(sme)
            metrics[f"{prefix}/n_sme"] = float(len(sme))
        names = sorted({n for _, v in items for n in v})
        for name in names:
            vals = [(_truthy(v[name]), r["sme"]) for r, v in items if name in v and v[name] is not None]
            if not vals:
                continue
            metrics[f"{prefix}/{name}"] = sum(ok for ok, _ in vals) / len(vals)
            both = [(ok, s) for ok, s in vals if s is not None]
            if both:
                metrics[f"{prefix}/{name}/agreement"] = sum(ok == s for ok, s in both) / len(both)
                metrics[f"{prefix}/{name}/n"] = float(len(both))
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--judge", default="true", help="false = code scorers only (no LLM judge calls)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    sys.path.insert(0, os.path.join(base, "eval"))
    import eval_datasets
    import mlflow
    import pandas as pd

    if eval_datasets.private_dir(base) is None:
        print("[reference] St. Jude's private ground truths not found — BioASQ only")
    judge = args.judge.strip().lower() == "true"
    rows = reference_rows(base)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        scored = list(pool.map(lambda r: score_row(r, judge), rows))
    metrics = summarize(rows, scored)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="mygpt-reference-baseline"):
        mlflow.set_tags({"mygpt.eval.kind": "reference", "mygpt.eval.judge": str(judge).lower()})
        mlflow.log_metrics(metrics)
        table = pd.DataFrame([
            {"use_case": r["use_case"], "config": r["config"], "key": r["key"], "question": r.get("question_id") or r.get("question_type"),
             "mygpt_answer": (r["outputs"] or "")[:500], "sme_correct": r["sme"], **{k: str(v) for k, v in s.items()}}
            for r, s in zip(rows, scored) if r["inputs"] is not None
        ])
        if not table.empty:
            mlflow.log_table(table, artifact_file="reference_rows.json")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.3f}")


if __name__ == "__main__":
    main()
