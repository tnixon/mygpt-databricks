"""Generate mygpt_benchmarks.lvdash.json (the AI/BI dashboard deployed by the root bundle).

Edit the SQL in dashboards/sql/ and the layout here, then re-run:

    python3 dashboards/build_dashboard.py

Every dataset is `sql/base.sql` (shared CTEs over the MLflow system tables) + `sql/<name>.sql`,
so the benchmark mapping lives in one place. Don't hand-edit the generated JSON.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "mygpt_benchmarks.lvdash.json")

DATASETS = {
    "scorecard": "Scorecard: Databricks agent vs MyGPT",
    "scorecard_long": "Scorecard (long form, for the grouped bar)",
    "robustness": "SJ Global: original vs reworded questions",
    "kinase_configs": "BlueSky: agent vs each MyGPT config",
    "scorer_trust": "Scorer agreement with St. Jude SMEs",
    "trend": "Headline metrics per full eval run",
    "runs": "Eval + reference runs",
}

GREEN, RED, AMBER = "#C8E6C9", "#FFCDD2", "#FFF3C4"


def dataset(name: str) -> dict:
    with open(os.path.join(HERE, "sql", "base.sql")) as f:
        base = f.read().rstrip("\n")
    with open(os.path.join(HERE, "sql", f"{name}.sql")) as f:
        body = f.read().rstrip("\n")
    return {"name": name, "displayName": DATASETS[name], "queryLines": [line + "\n" for line in (base + "\n" + body).split("\n")]}


def query(name: str, ds: str, fields: list[tuple[str, str]], disaggregated: bool = True) -> dict:
    return {"name": name, "query": {"datasetName": ds, "disaggregated": disaggregated,
                                    "fields": [{"name": n, "expression": e} for n, e in fields]}}


def cols(*names: str) -> list[tuple[str, str]]:
    return [(n, f"`{n}`") for n in names]


def text(name: str, md: str, x: int, y: int, w: int, h: int) -> dict:
    return {"widget": {"name": name, "textbox_spec": md}, "position": {"x": x, "y": y, "width": w, "height": h}}


def counter(name: str, ds: str, column: str, title: str, x: int, y: int) -> dict:
    field = f"sum({column})"
    return {
        "widget": {
            "name": name,
            "queries": [query("main_query", ds, [(field, f"SUM(`{column}`)")], disaggregated=False)],
            "spec": {"version": 2, "widgetType": "counter",
                     "encodings": {"value": {"fieldName": field, "displayName": title}},
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": 1, "height": 3},
    }


def chart(name: str, kind: str, ds: str, title: str, x_enc: dict, y_enc: dict, color: dict | None,
          pos: tuple[int, int, int, int], extra: dict | None = None) -> dict:
    fields = [x_enc["fieldName"], y_enc["fieldName"]] + ([color["fieldName"]] if color else [])
    encodings = {"x": x_enc, "y": y_enc}
    if color:
        encodings["color"] = color
    spec = {"version": 3, "widgetType": kind, "encodings": encodings, "frame": {"title": title, "showTitle": True}}
    spec.update(extra or {})
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, cols(*fields))], "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def cat(field: str, label: str) -> dict:
    return {"fieldName": field, "scale": {"type": "categorical"}, "displayName": label}


def num(field: str, label: str, lo: float | None = 0, hi: float | None = 1) -> dict:
    scale: dict = {"type": "quantitative"}
    if lo is not None and hi is not None:
        scale["domain"] = {"min": lo, "max": hi}
    return {"fieldName": field, "scale": scale, "displayName": label}


def filter_widget(name: str, field: str, title: str, datasets: list[str], pos: tuple[int, int, int, int],
                  multi: bool = False) -> dict:
    queries = [query(f"f_{ds}", ds, cols(field), disaggregated=False) for ds in datasets]
    x, y, w, h = pos
    return {
        "widget": {
            "name": name,
            "queries": queries,
            "spec": {"version": 2, "widgetType": "filter-multi-select" if multi else "filter-single-select",
                     "encodings": {"fields": [{"fieldName": field, "displayName": title, "queryName": q["name"]}
                                              for q in queries]},
                     "frame": {"title": title, "showTitle": True}},
        },
        "position": {"x": x, "y": y, "width": w, "height": h},
    }


def number_fmt(places: int) -> dict:
    return {"type": "number-plain", "abbreviation": "none", "decimalPlaces": {"type": "exact", "places": places}}


def rule(field: str, value: str, color: str) -> dict:
    return {"condition": {"operator": "=", "fieldName": field, "operand": {"type": "data-value", "value": value}},
            "backgroundColor": color}


def table(name: str, ds: str, title: str, columns: list[dict], pos: tuple[int, int, int, int], freeze: int = 0) -> dict:
    hidden = [c.pop("_also", []) for c in columns]
    fields = [c["fieldName"] for c in columns] + [f for extra in hidden for f in extra]
    spec = {"version": 2, "widgetType": "table", "encodings": {"columns": columns},
            "frame": {"title": title, "showTitle": True}, "data": {"queryName": "main_query"}}
    if freeze:
        spec["freezeUpToColumnNumber"] = freeze
    x, y, w, h = pos
    return {"widget": {"name": name, "queries": [query("main_query", ds, cols(*dict.fromkeys(fields)))], "spec": spec},
            "position": {"x": x, "y": y, "width": w, "height": h}}


def col(field: str, label: str, places: int | None = None, rules: list | None = None, template: str | None = None,
        also: list[str] | None = None) -> dict:
    c: dict = {"fieldName": field, "displayName": label}
    if places is not None:
        c["format"] = number_fmt(places)
    if template:
        c["formatTemplate"] = template
    if rules:
        c["style"] = {"type": "basic", "rules": rules}
    if also:
        c["_also"] = also
    return c


STATUS_RULES = [rule("status", "Better", GREEN), rule("status", "On par", AMBER), rule("status", "Worse", RED)]
GATE_RULES = [rule("gate_status", "Pass", GREEN), rule("gate_status", "Fail", RED)]

HEADER = """# MyGPT benchmarks: Databricks agent vs. MyGPT
**Databricks agent** = the `mygpt-rag-agent` endpoint as served at its latest **full** eval run (`mygpt_eval`, every row of each dataset).
**MyGPT** = the answers St. Jude's MyGPT recorded in *MyGPT-evaluations*, put through **the same scorers** (`mygpt_eval_reference`; nothing is regenerated).
Scores are accuracies (0-1). **Δ** = ours − MyGPT's best configuration; within ±0.03 counts as *on par*."""

HOW_TO_READ = """### How to read this
- **SJ Global (health policies, FR / ES):** MyGPT's bar is **St. Jude's SME grading** of MyGPT's `answer_1`. Our yes/no gold is *derived from* those answers (flipped where the SMEs marked them Incorrect), so MyGPT is not re-scored against it. The reworded questions (Q6-10) and the correctness judge have no MyGPT figure.
- **BlueSky (kinase literature):** MyGPT ran three configurations (5 or 15 context chunks, structured pre-prompt); the table shows their range and Δ is against the best. About half the kinase gold comes from a block of St. Jude's file that was shifted by one row and has been realigned (see *Robustness & trust*).
- **BioASQ (public benchmark, 163 questions over all 397 papers):** MyGPT's figure is its gpt-oss-20b run.
- Rows marked *(LLM judge)* use MLflow's correctness judge, which agrees with St. Jude's SMEs 75-84% of the time; treat ±0.05 differences there as noise. The code scorers agree 83-100%.
- **Gate** = the minimum the CI eval enforces for this metric (`eval/gates.yaml`); shown once a run has logged its gates."""

ROBUSTNESS_MD = """### Robustness: does the answer depend on the wording?
SJ Global questions Q6-10 repeat a base question **without its keyword hints**, and inherit the base question's yes/no gold. A large drop means the agent leans on the hinted vocabulary rather than the policy's meaning."""

TRUST_MD = """### Can the scorers be trusted?
Each gold scorer was run on MyGPT's own kinase answers and compared with St. Jude's SME Correct / Incorrect labels.
**≥ 0.85 agreement**: reliable enough to gate on. **< 0.80**: read as indicative only.
`sme` = gold as St. Jude recorded it; `sme_realigned` = gold repaired from the shifted block of their file (agreement there validates the realignment)."""

TREND_MD = """### Trend across agent versions
Every **full** eval run (smoke runs with `max_rows` > 0 are excluded). Gold metrics start at agent v12; the judge metrics go back to v9.
Use the filters to pick a dataset and metrics."""


def build() -> dict:
    p1 = [
        text("header", HEADER, 0, 0, 6, 3),
        counter("c_at_or_above", "scorecard", "is_at_or_above", "Benchmarks at or above MyGPT", 0, 3),
        counter("c_below", "scorecard", "is_below", "Benchmarks below MyGPT", 1, 3),
        counter("c_gates_passed", "scorecard", "gate_passed", "Gates passing", 2, 3),
        counter("c_gated", "scorecard", "gated", "Gated metrics", 3, 3),
        filter_widget("f_model", "model_alias", "Agent LLM", ["scorecard", "scorecard_long"], (4, 3, 2, 3)),
        table("t_scorecard", "scorecard", "Scorecard", [
            col("use_case_label", "Use case"),
            col("dataset_label", "Dataset"),
            col("benchmark", "Benchmark"),
            col("ours", "Databricks agent", 3),
            col("mygpt_range", "MyGPT"),
            col("delta", "Δ vs MyGPT best", 3, STATUS_RULES, "{{ delta_sign }}{{ @formatted }}", also=["delta_sign"]),
            col("status", "Status", rules=STATUS_RULES),
            col("gate", "Gate", 2),
            col("gate_status", "Gate result", rules=GATE_RULES),
            col("mygpt_configs", "MyGPT by config"),
            col("mygpt_basis", "MyGPT figure is"),
            col("agent_version", "Agent version"),
            col("model_alias", "LLM"),
        ], (0, 6, 6, 9), freeze=3),
        chart("b_compare", "bar", "scorecard_long", "Databricks agent vs MyGPT (best config)",
              num("score", "Score"), cat("line", "Benchmark"), cat("system", "System"), (0, 15, 3, 10),
              {"mark": {"layout": "group"}}),
        chart("b_delta", "bar", "scorecard", "Δ vs MyGPT (positive = Databricks agent ahead)",
              num("delta", "Δ", None, None), cat("line", "Benchmark"), cat("status", "Status"), (3, 15, 3, 10)),
        text("how_to_read", HOW_TO_READ, 0, 25, 6, 6),
    ]
    p2 = [
        text("robustness_md", ROBUSTNESS_MD, 0, 0, 3, 3),
        text("trust_md", TRUST_MD, 3, 0, 3, 3),
        chart("b_robustness", "bar", "robustness", "SJ Global yes/no: original vs reworded",
              cat("dataset_label", "Dataset"), num("yes_no_correct", "Yes/no correct"), cat("wording", "Question wording"),
              (0, 3, 3, 8), {"mark": {"layout": "group"}}),
        chart("b_trust", "bar", "scorer_trust", "Scorer agreement with SMEs, by MyGPT config",
              num("agreement", "Agreement"), cat("label", "Scorer (gold slice)"), cat("config", "MyGPT config"),
              (3, 3, 3, 8), {"mark": {"layout": "group"}}),
        chart("b_kinase", "bar", "kinase_configs", "BlueSky per question: Databricks agent vs each MyGPT config",
              cat("question", "Question"), num("score", "Score"), cat("system", "System"),
              (0, 11, 6, 8), {"mark": {"layout": "group"}}),
    ]
    p3 = [
        text("trend_md", TREND_MD, 0, 0, 6, 2),
        filter_widget("f_trend_dataset", "dataset_label", "Dataset", ["trend"], (0, 2, 2, 2), multi=True),
        filter_widget("f_trend_metric", "metric_label", "Metric", ["trend"], (2, 2, 2, 2), multi=True),
        filter_widget("f_trend_model", "model_alias", "Agent LLM", ["trend"], (4, 2, 2, 2)),
        chart("l_trend", "line", "trend", "Headline metrics by run",
              {"fieldName": "run_time", "scale": {"type": "temporal"}, "displayName": "Run time"},
              num("score", "Score"), cat("series", "Dataset | metric"), (0, 4, 6, 9)),
        table("t_runs", "runs", "Runs", [
            col("run_time", "Run time"),
            col("run_name", "Run"),
            col("kind", "Kind"),
            col("model_alias", "LLM"),
            col("served_model", "Served model"),
            col("max_rows", "max_rows"),
            col("run_url", "Open in MLflow"),
        ], (0, 13, 6, 8)),
    ]
    return {
        "datasets": [dataset(n) for n in DATASETS],
        "pages": [
            {"name": "scorecard", "displayName": "Scorecard", "pageType": "PAGE_TYPE_CANVAS", "layout": p1},
            {"name": "robustness_trust", "displayName": "Robustness & trust", "pageType": "PAGE_TYPE_CANVAS", "layout": p2},
            {"name": "trend", "displayName": "Trend", "pageType": "PAGE_TYPE_CANVAS", "layout": p3},
        ],
    }


if __name__ == "__main__":
    with open(OUT, "w") as f:
        json.dump(build(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {OUT}")
