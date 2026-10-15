# eval — MLflow GenAI evaluation (Phase 2)

Measurement-first evaluation of the `mygpt-rag-agent` endpoint with **MLflow 3**
`mlflow.genai.evaluate()`, seeded from St. Jude's **MyGPT-evaluations** protocols. Rows are
graded by LLM judges + code scorers against per-row guidelines and, where it exists, **gold**:
St. Jude's private SME ground truths (SJ Global, BlueSky) and the public BioASQ benchmark.

| Use case | Source | UC eval dataset | Rows | Gold |
|---|---|---|---|---|
| SJ Global | `health_policies/` (FR) | `structural_biology.mygpt.eval_health_policies_fr` | 21 questions × 10 policies = 210 | 160 SME + 50 inherited (yes/no) |
| SJ Global | `health_policies/` (ES) | `structural_biology.mygpt.eval_health_policies_es` | 210 | 160 SME + 50 inherited |
| BlueSky | `kinase_literature/` | `structural_biology.mygpt.eval_kinase_literature` | 3 questions × 94 paper/kinase pairs = 282 | SME ground truth on most rows |
| BioASQ | `BioASQ/` (public) | `structural_biology.mygpt.eval_bioasq` | 163 (76 yes/no + 87 factoid) | benchmark `exact_answer` / `ideal_answer` |

For SJ Global and BlueSky, each row's `inputs` is the exact agent request (question +
`custom_inputs.corpus` + `focus.documents`), so every question is asked of one document,
which is MyGPT's protocol. BioASQ follows MyGPT's BioASQ protocol instead: library-wide retrieval over the
`biomed` corpus (all 397 BioASQ PDFs), no document focus, `k = 15`.
`expectations.guidelines` carries per-row guidelines for `ExpectationsGuidelines`.

Everything lives in one shared experiment, **`/Shared/mygpt/mygpt-rag-agent`** (DAB
resource): the agent's LoggedModels, these eval runs (linked via `model_id`), the reference
baseline run, and the endpoint's production traces + monitoring scorers.

## Gold
| Key in `expectations` | Used by | Where it comes from |
|---|---|---|
| `expected_label` (yes / no) | `yes_no_correct` | SJ Global: MyGPT's `answer_1` yes/no, **flipped when the SMEs marked it Incorrect**. BioASQ: `exact_answer` of yes/no questions. |
| `expected_values` | `expression_system_correct` (Q11), `figure_ref_match` (Q17), `factoid_match` (BioASQ) | Kinase `ground_truth` split on `;`. BioASQ factoid synonyms. |
| `expected_response` | `gold_correctness` | SJ Global: `answer_1` when Correct. BioASQ: the first `ideal_answer`. |
| `expected_facts` | `gold_correctness` | Kinase Q16v2: the gold assay *category* ("protein kinase assay"). |

`tags.gold` records provenance, and `run_eval` reports every gold scorer per slice
(`<scorer>/<gold>/mean`):
- `sme`: labelled by St. Jude.
- `inherited`: SJ Global Q6–10. These variants drop the keyword hints and reuse the base
  question's **yes/no label only**. They're marked `gold_scope: label_only`, so the
  correctness judge skips them: the base question's reference summary was written for the
  hinted question.
- `sme_realigned`: see below.
- `benchmark`: BioASQ.
- `none`: no gold. The gold scorers skip these rows.

**Changing gold:** `merge_records` merges expectation keys into existing rows. It doesn't
replace them, so deleting a key in a builder leaves the old value in the UC table. To retract
gold, mark the row instead (as `gold_scope: label_only` does), or rebuild the dataset.

**Data issue in St. Jude's kinase file (reported to them):**
- In ids 129–262, the `question_id` / `ground_truth` cells sit one row too high relative to the
  question text, MyGPT's answers and the SME labels.
- `kinase_gold_rows` keys rows by question **text** and takes the gold from the previous row
  inside that block. Rows repaired this way are tagged `sme_realigned`.
- A shape check drops any gold that doesn't fit its question (figure refs only on Q17). Row 129
  loses its gold at the block boundary.
- Ask St. Jude for a corrected file. Once it arrives, the realignment no longer triggers.

### Private data handling
- The private CSVs (`reference/MyGPT-evaluations-private-main/`) are gitignored and excluded
  from the bundle sync.
- `make eval-private PROFILE=<p>` copies them to the **`eval_private`** UC volume. It has no
  grants, so only its owner (the deployer, who is the eval jobs' run-as identity) and
  metastore admins can read it.
- Without the volume, the builders fall back to guidelines-only rows for SJ Global and BlueSky.
- **The eval-dataset tables then contain the gold.** Keep SELECT on `eval_*` limited to the
  project team, and check the schema-level grants on `structural_biology.mygpt`.
- The unit tests use synthetic CSVs. Tests that read the real files only assert structure and
  are skipped when the folder is absent.

## Files
| File | What |
|---|---|
| `eval_datasets.py` | Pure record builders: submodule CSVs + gold joins (unit-tested) |
| `build_datasets.py` | Job: merge records into the UC eval datasets (idempotent upsert) |
| `scorers.py` | Judges + code scorers + gold scorers, and the per-use-case scorer sets |
| `run_eval.py` | Job: one dataset × LLM alias → eval run, AUROC, gold-provenance breakdown, coding matrices, optional gate |
| `reference_baseline.py` | Job: MyGPT's own recorded answers through our gold scorers → parity baseline + judge agreement |
| `gates.yaml` | Minimum `<metric>/mean` per use case (judge/gold thresholds set after baseline) |
| `monitor.py` | Job: register the shared scorers on live app traffic |
| `test_scorers.py` | `make eval-test` |

## Scorers
- **Judges:** `RetrievalGroundedness`, `RetrievalRelevance`, `RelevanceToQuery`, `Safety`,
  `ExpectationsGuidelines`, `mygpt_grounding` (Guidelines). These read the retrieved
  context from the agent's RETRIEVER span (Document-shaped).
- **Shared:** `retrieval_ok`, `mygpt_trust` (QRS / ARS / HI as numeric feedback),
  `focus_adherence` (not for BioASQ, which has no focus).
- **SJ Global:** `yes_no_format`, `answer_label` → logged as the `coding_matrix.json`
  table (policy × question → yes / no); `yes_no_correct` and `gold_correctness` →
  `coding_matrix_vs_gold.json` (each cell beside its gold).
- **BlueSky:** `js_array_valid`, `figure_ref_format` (Q17), `kinase_in_context` +
  `no_match_consistent`, `vocab_compliance` (Q11, once the vocabulary lands);
  gold: `expression_system_correct` (Q11), `figure_ref_match` + `figure_ref_recall` (Q17),
  `gold_correctness` (Q16v2).
- **BioASQ:** `yes_no_correct` (yes/no questions), `factoid_match` (any gold synonym in the
  answer, which is BioASQ's lenient accuracy), `gold_correctness`.
- **Post-run metric:** `hi_eq_vs_groundedness_auroc`, which measures how well MyGPT's
  equation-HI ranks the rows the groundedness judge marks ungrounded.

## MyGPT reference baseline (`mygpt_eval_reference`)
This job scores MyGPT's published answers with **our** gold scorers. It makes no agent
endpoint calls. It logs the following to one run, `mygpt-reference-baseline`:
- `mygpt_ref/<use_case>/<config>/sme_accuracy`: St. Jude's grading of MyGPT. This is the
  parity bar for our agent.
- `mygpt_ref/<use_case>/<config>/<scorer>` and `…/<scorer>/agreement`: our scorer's
  accuracy on the same answers, and how often it agrees with the SMEs. Use the agreement to
  decide which gold scorers are trustworthy enough to gate on.

| Source | Configs | Reports |
|---|---|---|
| SJ Global | `answer_1` | SME accuracy only. Our policy gold is derived from these answers, so agreement would be circular. |
| BlueSky | `ctx5`, `ctx15`, `strpre` | SME accuracy + agreement for `expression_system_correct`, `figure_ref_match`, `gold_correctness` |
| BioASQ | `gpt-oss-20b` | accuracy only (no SME labels) |

## Benchmarks dashboard
**"[mygpt] Benchmarks — MyGPT vs Databricks agent"** is an AI/BI dashboard, deployed with
the root bundle (`resources.dashboards.mygpt_benchmarks`). It queries
`system.mlflow.runs_latest` for this experiment directly, so it has no tables or jobs of its own.
It refreshes on its own after every `mygpt_eval` / `mygpt_eval_reference` run.
- **Scorecard:** each benchmark line is our latest **full** run (`max_rows = 0`) per dataset ×
  LLM, set against MyGPT's figure. Δ is measured against MyGPT's best config; ±0.03 counts as on
  par. Each row also shows its gate and the gate result. `run_eval` logs the gates in force as
  `gate/<metric>` params, so runs from before 2026-10-15 show none.
- **Robustness & trust:** SJ Global original vs reworded questions; each scorer's agreement
  with the SMEs, by MyGPT config and gold slice; and BlueSky per question vs each MyGPT config.
- **Trend:** headline gold and judge metrics for every full run, by agent version.

To change it, edit `dashboards/sql/*.sql` and the layout in `dashboards/build_dashboard.py`, then
run `python3 dashboards/build_dashboard.py` and `bundle deploy`. `sql/base.sql` holds the
benchmark ↔ MyGPT-metric mapping. Don't hand-edit the generated `.lvdash.json`.

## Run
```bash
make eval-test                                                  # local unit tests
databricks bundle deploy -p mygpt --var warehouse_id=<id>        # first: creates the eval_private volume
make eval-corpus PROFILE=mygpt [KINASE_PDFS=<dir>]               # upload eval corpora (incl. BioASQ -> biomed)
make eval-private PROFILE=mygpt                                  # private gold -> eval_private volume
databricks bundle run mygpt_etl -p mygpt --var warehouse_id=<id> # incremental parse + MERGE + index sync
databricks bundle run mygpt_deploy_agent -p mygpt
databricks bundle run mygpt_eval_build_datasets -p mygpt
databricks bundle run mygpt_eval_reference -p mygpt              # MyGPT parity + judge agreement
databricks bundle run mygpt_eval -p mygpt --params max_rows=5    # smoke (default matrix)
databricks bundle run mygpt_eval -p mygpt --params gate=true     # full + gates
databricks bundle run mygpt_eval_monitor -p mygpt
```
`--params` is CSV-parsed, so a JSON `matrix` override must go through `jobs run-now`, e.g.
for the model comparison:
```bash
databricks jobs run-now -p mygpt --json '{"job_id": <mygpt_eval job id>, "job_parameters":
  {"matrix": "[{\"dataset\":\"health_policies_fr\",\"model\":\"gptoss\"}]"}}'
```

## Blocked on St. Jude (BlueSky)
- `kinase_literature/inputs/expression_systems_list.csv`, the Q11 vocabulary that
  `collect_answers.py` expects. Re-run `mygpt_eval_build_datasets` once it lands.
- A corrected `Ground truths Kinases Literature.csv` (the shifted ids 129–262 block, above).

## Deferred
- **PubMedQA:** no PDFs. They're paywalled, and St. Jude used institutional access.
- **QRS-ARS-cutoff:** 43 of its 54 rows are PubMedQA. So `embedding_model_cutoffs`
  recalibration is still open.
- **Open-RAG-Bench:** has gold, but its 170 arXiv PDFs aren't downloaded.
- **Judge alignment:** MLflow `align()` on the SME labels.
- **EN↔FR/ES translation:** Phase 6.
