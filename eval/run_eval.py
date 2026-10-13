"""Run one Phase-2 evaluation: a UC eval dataset x an LLM alias against mygpt-rag-agent.

DABs job: mygpt_eval (one for_each iteration per dataset x model). Each run:
  * calls the live agent endpoint, which returns its full trace (RETRIEVER span
    included); the trace is copied into the eval run for the judges to read;
  * scores with the use case's scorer set (eval/scorers.py);
  * links the run to the served agent's LoggedModel (model_id), so results line up
    against model versions in the shared experiment;
  * logs post-run metrics/artifacts — the HI-vs-groundedness AUROC (do MyGPT's own
    trust scores agree with the judges?), gold-scorer means split by gold provenance
    (SME / inherited / realigned), and, for SJ Global, the policy x question coding
    matrix plus its comparison with the SME gold;
  * optionally gates on eval/gates.yaml (non-zero exit on a miss).

Usage:
    python eval/run_eval.py --dataset health_policies_fr --model sonnet [--max-rows 20] [--gate true]
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def _base_dir(base_dir: str) -> str:
    if base_dir:
        return base_dir
    try:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except NameError:
        return os.getcwd()


def served_model_id(endpoint: str) -> tuple[str | None, str | None]:
    """(LoggedModel id, 'name/version') of the UC model version the endpoint serves."""
    try:
        from databricks.sdk import WorkspaceClient
        from mlflow import MlflowClient

        ep = WorkspaceClient().serving_endpoints.get(endpoint)
        # agents.deploy keeps every prior version as a served entity (at 0% traffic) —
        # the version under test is the one the traffic routes send requests to.
        routes = ep.config.traffic_config.routes if ep.config.traffic_config else []
        live = max(routes, key=lambda r: r.traffic_percentage or 0).served_entity_name if routes else None
        ent = next(e for e in ep.config.served_entities if e.entity_version and (live is None or e.name == live))
        mv = MlflowClient(registry_uri="databricks-uc").get_model_version(ent.entity_name, ent.entity_version)
        return getattr(mv, "model_id", None), f"{ent.entity_name}/{ent.entity_version}"
    except Exception as exc:  # noqa: BLE001
        print(f"[run_eval] could not resolve served model version: {type(exc).__name__}: {exc}")
        return None, None


def persisted_results(run_id: str, require: tuple[str, ...] = (), timeout_s: int = 300):
    """Per-row results as persisted on the run's traces ([trace_id, request, <name>/value, ...]).
    Used for the post-run metrics instead of EvaluationResult.result_df, which can be
    missing assessments that are still being logged when evaluate() returns.

    Assessments are persisted asynchronously, so poll until every row has a value for each
    `require`d assessment (or the timeout passes — then return what's there)."""
    import time

    import mlflow
    import pandas as pd
    from mlflow.genai.utils.trace_utils import _get_assessment_values

    deadline = time.time() + timeout_s
    while True:
        df = mlflow.search_traces(run_id=run_id, return_type="pandas")
        if df is None or df.empty:
            rows = None
        else:
            values = df["assessments"].apply(lambda a: _get_assessment_values(a, run_id)).apply(pd.Series)
            rows = pd.concat([df[["trace_id", "request"]], values], axis=1)
        complete = rows is not None and all(
            f"{name}/value" in rows and rows[f"{name}/value"].notna().all() for name in require
        )
        if complete or time.time() > deadline:
            return rows
        time.sleep(15)


def endpoint_predict_fn(endpoint: str, model_alias: str):
    """predict_fn for the agent endpoint that ALWAYS copies the endpoint's returned trace
    into the eval run.

    mlflow.genai.to_predict_fn skips the copy when the endpoint already dual-writes its
    traces to the current experiment (our setup: the agent's MLFLOW_EXPERIMENT_ID is the
    shared experiment) and then never hands that trace to evaluate(), which falls back to
    a root-only trace with no RETRIEVER span (RetrievalGroundedness / RetrievalRelevance
    then fail with "No retrieval context found"). This mirrors to_predict_fn's copy branch.
    """
    from mlflow.deployments import get_deploy_client
    from mlflow.tracing.constant import DATABRICKS_OPTIONS_KEY, DATABRICKS_OUTPUT_KEY, RETURN_TRACE_OPTION_KEY
    from mlflow.tracing.utils.copy import copy_trace_to_experiment

    client = get_deploy_client("databricks")

    def predict_fn(input, custom_inputs=None):  # noqa: A002  (mirrors the agent request keys)
        # Dataset rows are model-agnostic; the LLM under test is chosen per run.
        payload = {
            "input": input,
            "custom_inputs": {**(custom_inputs or {}), "model": model_alias},
            DATABRICKS_OPTIONS_KEY: {RETURN_TRACE_OPTION_KEY: True},
        }
        result = client.predict(endpoint=endpoint, inputs=payload)
        trace = (result.pop(DATABRICKS_OUTPUT_KEY, None) or {}).get("trace")
        if trace:
            try:
                copy_trace_to_experiment(trace)
            except Exception as exc:  # noqa: BLE001
                # evaluate()'s up-front trace check runs predict_fn with no-op spans, where
                # the copy can't record — as in to_predict_fn, never fail the prediction.
                print(f"[run_eval] could not copy endpoint trace: {type(exc).__name__}: {exc}")
        return result

    return predict_fn


def _request_of(row) -> dict:
    req = row.get("request")
    if isinstance(req, str):
        try:
            req = json.loads(req)
        except json.JSONDecodeError:
            return {}
    if not isinstance(req, dict):
        return {}
    # The agent endpoint's root span records its argument as {"request": <ResponsesAgentRequest>}.
    if "custom_inputs" not in req and isinstance(req.get("request"), dict):
        req = req["request"]
    return req


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    """Rank-based AUROC (Mann-Whitney U) — no sklearn dependency in the eval env."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def hi_vs_groundedness(df) -> float | None:
    """How well MyGPT's equation-HI ranks the rows the groundedness judge marks ungrounded."""
    if df is None or "hi_eq/value" not in df or "retrieval_groundedness/value" not in df:
        return None
    rows = df[["hi_eq/value", "retrieval_groundedness/value"]].dropna()
    return auroc(
        [float(v) for v in rows["hi_eq/value"]],
        [str(v).lower() == "no" for v in rows["retrieval_groundedness/value"]],
    )


def _record_key(req: dict) -> tuple[str | None, str | None]:
    """(question text, focused document) — the identity of a dataset row in a trace."""
    ci = req.get("custom_inputs") or {}
    question = next((m.get("content") for m in req.get("input", []) if m.get("role") == "user"), None)
    docs = (ci.get("focus") or {}).get("documents") or [None]
    return question, docs[0]


def _records_by_key(records: list[dict]) -> dict:
    return {_record_key(r["inputs"]): r for r in records}


# Gold scorers whose means are also reported per gold provenance (tags.gold).
GOLD_SCORERS = ("yes_no_correct", "expression_system_correct", "figure_ref_match", "factoid_match", "gold_correctness")


def gold_breakdown(df, records: list[dict]) -> dict[str, float]:
    """`<scorer>/<gold>/mean` for each gold scorer and provenance, e.g. yes_no_correct on the
    SME-labelled base questions vs the variants that inherit their gold."""
    if df is None or df.empty:
        return {}
    by_key = _records_by_key(records)
    gold = [((by_key.get(_record_key(_request_of(row))) or {}).get("tags") or {}).get("gold") for _, row in df.iterrows()]
    out = {}
    for name in GOLD_SCORERS:
        col = f"{name}/value"
        if col not in df:
            continue
        groups: dict[str, list[float]] = {}
        for g, v in zip(gold, df[col]):
            if g and v is not None and str(v) not in ("nan", "None"):
                groups.setdefault(g, []).append(1.0 if str(v).lower() in ("true", "yes", "1", "1.0") else 0.0)
        for g, vals in groups.items():
            out[f"{name}/{g}/mean"] = sum(vals) / len(vals)
    return out


def coding_matrix_vs_gold(df, records: list[dict]):
    """SJ Global: each (document, question) answer label beside its gold label, for the
    rows that have gold — the per-cell view behind yes_no_correct."""
    import pandas as pd

    if df is None or "answer_label/value" not in df:
        return None
    by_key = _records_by_key(records)
    cells = []
    for _, row in df.iterrows():
        rec = by_key.get(_record_key(_request_of(row)))
        if not rec or not rec["expectations"].get("expected_label"):
            continue
        tags = rec["tags"]
        cells.append({
            "document": tags["document"], "question_key": tags["question_key"], "variant": tags["variant"],
            "gold_source": tags["gold"], "answer": row["answer_label/value"],
            "gold": rec["expectations"]["expected_label"],
            "correct": row["answer_label/value"] == rec["expectations"]["expected_label"],
        })
    return pd.DataFrame(cells) if cells else None


def coding_matrix(df, records: list[dict]):
    """SJ Global deliverable: policy document x question_key -> yes / no / unparseable."""
    import pandas as pd

    if df is None or "answer_label/value" not in df:
        return None
    key_of = {r["inputs"]["input"][0]["content"]: r["tags"]["question_key"] + ("*" if r["tags"]["variant"] == "yes" else "")
              for r in records}
    cells = []
    for _, row in df.iterrows():
        req = _request_of(row)
        ci = req.get("custom_inputs") or {}
        question = next((m.get("content") for m in req.get("input", []) if m.get("role") == "user"), None)
        docs = (ci.get("focus") or {}).get("documents") or [None]
        cells.append({"document": docs[0], "question": key_of.get(question, question), "label": row["answer_label/value"]})
    if not cells:
        return None
    return pd.DataFrame(cells).pivot_table(index="document", columns="question", values="label", aggfunc="first")


def check_gates(metrics: dict, use_case: str, gates_file: str) -> list[str]:
    import yaml

    with open(gates_file) as f:
        gates = yaml.safe_load(f) or {}
    failures = []
    for metric, minimum in {**(gates.get("shared") or {}), **(gates.get(use_case) or {})}.items():
        if minimum is None:
            continue
        value = metrics.get(metric)
        if value is None:
            failures.append(f"{metric}: missing (gate >= {minimum})")
        elif value < minimum:
            failures.append(f"{metric}: {value:.3f} < {minimum}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="health_policies_fr | health_policies_es | kinase_literature | bioasq")
    ap.add_argument("--model", default="sonnet", help="agent LLM alias: sonnet | llama | gptoss")
    ap.add_argument("--endpoint", default="mygpt-rag-agent")
    ap.add_argument("--catalog", default="structural_biology")
    ap.add_argument("--schema", default="mygpt")
    ap.add_argument("--experiment", default="/Shared/mygpt/mygpt-rag-agent")
    ap.add_argument("--max-rows", type=int, default=0, help="smoke-test on the first N rows (0 = all)")
    ap.add_argument("--gate", default="false", help="true = exit non-zero if eval/gates.yaml isn't met")
    # The agent endpoint is CPU Small (~4 concurrent requests, ~6s/row warm): keep eval
    # parallelism near that and allow for queueing / cold starts rather than timing out.
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--predict-timeout", type=int, default=300, help="seconds per endpoint call")
    ap.add_argument("--git-sha", default="")
    ap.add_argument("--base-dir", default="")
    args = ap.parse_args()

    base = _base_dir(args.base_dir)
    sys.path.insert(0, os.path.join(base, "eval"))
    # Must be set before mlflow reads them (at predict / evaluate time).
    os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(args.workers)
    os.environ["MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT"] = str(args.predict_timeout)
    import eval_datasets
    import mlflow
    import scorers as sc
    from mlflow.genai import datasets

    spec = eval_datasets.specs(args.catalog, args.schema)[args.dataset]
    mlflow.set_experiment(args.experiment)
    data = datasets.get_dataset(name=spec.uc_table)
    if args.max_rows:
        data = data.to_df()[["inputs", "expectations"]].head(args.max_rows)

    predict_fn = endpoint_predict_fn(args.endpoint, args.model)

    vocab = eval_datasets.expression_systems(base) if spec.use_case == "bluesky" else None
    model_id, served = served_model_id(args.endpoint)

    with mlflow.start_run(run_name=f"{args.dataset}-{args.model}") as run:
        mlflow.set_tags({
            "mygpt.eval.dataset": spec.uc_table, "mygpt.eval.use_case": spec.use_case,
            "mygpt.eval.model_alias": args.model, "mygpt.eval.endpoint": args.endpoint,
            "mygpt.eval.served_model": served or "", "mygpt.eval.git_sha": args.git_sha,
            "mygpt.eval.max_rows": str(args.max_rows),
        })
        result = mlflow.genai.evaluate(
            data=data, predict_fn=predict_fn, scorers=sc.scorers_for(spec.use_case, vocab), model_id=model_id,
        )
        metrics = dict(result.metrics)
        needed = ("hi_eq", "retrieval_groundedness") + (("answer_label",) if spec.use_case == "sj_global" else ())
        rows = persisted_results(run.info.run_id, require=needed)

        if (auc := hi_vs_groundedness(rows)) is not None:
            mlflow.log_metric("hi_eq_vs_groundedness_auroc", auc)
            metrics["hi_eq_vs_groundedness_auroc"] = auc
        records = eval_datasets.build_records(args.dataset, base)
        if breakdown := gold_breakdown(rows, records):
            mlflow.log_metrics(breakdown)
            metrics.update(breakdown)
        if spec.use_case == "sj_global":
            matrix = coding_matrix(rows, records)
            if matrix is not None:
                mlflow.log_table(matrix.reset_index(), artifact_file="coding_matrix.json")
            vs_gold = coding_matrix_vs_gold(rows, records)
            if vs_gold is not None:
                mlflow.log_table(vs_gold, artifact_file="coding_matrix_vs_gold.json")

    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.3f}" if isinstance(metrics[k], float) else f"  {k}: {metrics[k]}")

    if args.gate.strip().lower() == "true":
        failures = check_gates(metrics, spec.use_case, os.path.join(base, "eval", "gates.yaml"))
        if failures:
            print("GATE FAILED:\n  " + "\n  ".join(failures))
            sys.exit(1)
        print("GATE PASSED")


if __name__ == "__main__":
    main()
