# eval — MLflow GenAI evaluation (Phase 2)

Measurement-first evaluation of the `mygpt-rag-agent` endpoint with **MLflow 3**
`mlflow.genai.evaluate()`, seeded from St. Jude's **MyGPT-evaluations** protocols for the two
real use cases. Guidelines-only for now (no gold answers): LLM judges + code scorers.

| Use case | Source | UC eval dataset | Rows |
|---|---|---|---|
| SJ Global | `health_policies/` (FR) | `structural_biology.mygpt.eval_health_policies_fr` | 21 questions × 10 policies = 210 |
| SJ Global | `health_policies/` (ES) | `structural_biology.mygpt.eval_health_policies_es` | 210 |
| BlueSky | `kinase_literature/` | `structural_biology.mygpt.eval_kinase_literature` | 3 questions × 94 paper/kinase pairs = 282 |

Each row's `inputs` is the exact agent request (question + `custom_inputs.corpus` +
`focus.documents`), so every question is asked of one document — MyGPT's protocol.
`expectations.guidelines` carries per-row guidelines for `ExpectationsGuidelines`.

Everything lives in one shared experiment, **`/Shared/mygpt/mygpt-rag-agent`** (DAB
resource): the agent's LoggedModels, these eval runs (linked via `model_id`), and the
endpoint's production traces + monitoring scorers.

## Files
| File | What |
|---|---|
| `eval_datasets.py` | Pure record builders from the submodule CSVs (unit-tested) |
| `build_datasets.py` | Job: merge records into the UC eval datasets (idempotent upsert) |
| `scorers.py` | Judges + code scorers, and the per-use-case scorer sets |
| `run_eval.py` | Job: one dataset × LLM alias → eval run, AUROC + coding matrix, optional gate |
| `gates.yaml` | Minimum `<metric>/mean` per use case (judge thresholds set after baseline) |
| `monitor.py` | Job: register the shared scorers on live app traffic |
| `test_scorers.py` | `make eval-test` |

## Scorers
- **Judges:** `RetrievalGroundedness`, `RetrievalRelevance`, `RelevanceToQuery`, `Safety`,
  `ExpectationsGuidelines`, `mygpt_grounding` (Guidelines). These read the retrieved
  context from the agent's RETRIEVER span (Document-shaped).
- **Shared:** `retrieval_ok`, `mygpt_trust` (QRS / ARS / HI as numeric feedback),
  `focus_adherence`.
- **SJ Global:** `yes_no_format`, `answer_label` → logged as the `coding_matrix.json`
  table (policy × question → yes / no).
- **BlueSky:** `js_array_valid`, `figure_ref_format` (Q17), `kinase_in_context` +
  `no_match_consistent`, `vocab_compliance` (Q11, once the vocabulary lands).
- **Post-run metric:** `hi_eq_vs_groundedness_auroc`, which measures how well MyGPT's
  equation-HI ranks the rows the groundedness judge marks ungrounded.

## Run
```bash
make eval-test                                                  # local unit tests
make eval-corpus PROFILE=mygpt [KINASE_PDFS=<dir>]               # upload eval corpora to the docs volume
databricks bundle deploy -p mygpt --var warehouse_id=<id>
databricks bundle run mygpt_etl -p mygpt --var warehouse_id=<id> # + recreate/sync the index (02 SQL tail)
databricks bundle run mygpt_deploy_agent -p mygpt
databricks bundle run mygpt_eval_build_datasets -p mygpt
databricks bundle run mygpt_eval -p mygpt --params max_rows=5    # smoke
databricks bundle run mygpt_eval -p mygpt --params gate=true     # full + gates
databricks bundle run mygpt_eval_monitor -p mygpt
```
Override the matrix for the model comparison, e.g.
`--params 'matrix=[{"dataset":"health_policies_fr","model":"gptoss"}]'`.

## Blocked on St. Jude (BlueSky)
- The kinase PDFs, named `<pubmed_id>.pdf` (or a `pdf_name` column in `kinase_pubmed_pairs.csv`).
- `kinase_literature/inputs/expression_systems_list.csv`, the Q11 vocabulary that
  `collect_answers.py` expects. Re-run `mygpt_eval_build_datasets` once it lands.

## Deferred
Public benchmarks (BioASQ, PubMedQA, Open-RAG-Bench); SME labeling / judge alignment and
gold expectations; `embedding_model_cutoffs` recalibration (QRS-ARS-cutoff);
EN↔FR/ES translation (Phase 6).
