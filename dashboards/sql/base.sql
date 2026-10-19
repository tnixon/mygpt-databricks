-- Shared CTEs, prepended to every dataset query by build_dashboard.py.
-- Source: MLflow's system tables (runs + their logged metrics) for the shared experiment
-- (its name is filled in by build_dashboard.py --experiment, i.e. var.experiment_path).
WITH runs AS (
  SELECT r.run_id, r.experiment_id, r.run_name, r.start_time, r.tags, r.params, r.aggregated_metrics
  FROM system.mlflow.runs_latest r
  JOIN system.mlflow.experiments_latest e
    ON r.workspace_id = e.workspace_id AND r.experiment_id = e.experiment_id
  WHERE e.name = '{experiment}' AND e.delete_time IS NULL
    AND r.delete_time IS NULL AND r.status = 'FINISHED'
),
metrics AS (
  SELECT run_id, run_name, start_time, tags, params, m.metric_name AS metric, m.latest_value AS value
  FROM runs LATERAL VIEW explode(aggregated_metrics) x AS m
),
-- One row per benchmark line of the comparison. `mygpt_rx` matches the reference run's
-- metric(s) for the same scorer: one per MyGPT config (3 for kinase), NULL = no MyGPT figure.
benchmarks AS (
  SELECT * FROM VALUES
    (1, 'sj_global', 'SJ Global', 'Yes/no vs SME gold (original wording)', 'yes_no_correct/sme/mean',
        '^mygpt_ref/sj_global/answer_1/sme_accuracy$', 'St. Jude SME accuracy of MyGPT answer_1'),
    (2, 'sj_global', 'SJ Global', 'Yes/no on reworded questions (Q6-10)', 'yes_no_correct/inherited/mean',
        CAST(NULL AS STRING), 'no MyGPT figure (variants were not SME-graded)'),
    (3, 'sj_global', 'SJ Global', 'Answer correctness (LLM judge)', 'gold_correctness/mean',
        CAST(NULL AS STRING), 'no MyGPT figure (gold is built from the MyGPT answers SMEs marked Correct)'),
    (4, 'bluesky', 'BlueSky kinase', 'Q11 expression system', 'expression_system_correct/mean',
        '^mygpt_ref/bluesky/(ctx5|ctx15|strpre)/expression_system_correct$', 'MyGPT ctx5 / ctx15 / strpre, same scorer'),
    (5, 'bluesky', 'BlueSky kinase', 'Q17 figure references', 'figure_ref_match/mean',
        '^mygpt_ref/bluesky/(ctx5|ctx15|strpre)/figure_ref_match$', 'MyGPT ctx5 / ctx15 / strpre, same scorer'),
    (6, 'bluesky', 'BlueSky kinase', 'Q16v2 assay type (LLM judge)', 'gold_correctness/mean',
        '^mygpt_ref/bluesky/(ctx5|ctx15|strpre)/gold_correctness$', 'MyGPT ctx5 / ctx15 / strpre, same judge'),
    (7, 'bioasq', 'BioASQ', 'Yes/no questions (76)', 'yes_no_correct/mean',
        '^mygpt_ref/bioasq/gpt-oss-20b/yes_no_correct$', 'MyGPT (gpt-oss-20b), same scorer'),
    (8, 'bioasq', 'BioASQ', 'Factoid questions (87)', 'factoid_match/mean',
        '^mygpt_ref/bioasq/gpt-oss-20b/factoid_match$', 'MyGPT (gpt-oss-20b), same scorer'),
    (9, 'bioasq', 'BioASQ', 'Answer correctness (LLM judge)', 'gold_correctness/mean',
        '^mygpt_ref/bioasq/gpt-oss-20b/gold_correctness$', 'MyGPT (gpt-oss-20b), same judge')
  AS b(sort, use_case, use_case_label, benchmark, ours_metric, mygpt_rx, mygpt_basis)
),
-- Our agent: full-dataset eval runs (max_rows = 0); rk = 1 is the latest per dataset x model.
ours AS (
  SELECT run_id, start_time, metric, value, params,
         tags['mygpt.eval.use_case'] AS use_case,
         tags['mygpt.eval.model_alias'] AS model_alias,
         CASE regexp_extract(tags['mygpt.eval.dataset'], '([a-z_]+)$', 1)
           WHEN 'eval_health_policies_fr' THEN 'SJ Global (FR)'
           WHEN 'eval_health_policies_es' THEN 'SJ Global (ES)'
           WHEN 'eval_kinase_literature' THEN 'BlueSky kinase'
           WHEN 'eval_bioasq' THEN 'BioASQ'
           ELSE tags['mygpt.eval.dataset'] END AS dataset_label,
         CAST(regexp_extract(tags['mygpt.eval.served_model'], '/([0-9]+)$', 1) AS INT) AS agent_version,
         dense_rank() OVER (PARTITION BY tags['mygpt.eval.dataset'], tags['mygpt.eval.model_alias']
                            ORDER BY start_time DESC) AS rk
  FROM metrics
  WHERE tags['mygpt.eval.use_case'] IS NOT NULL AND tags['mygpt.eval.max_rows'] = '0'
),
-- MyGPT: its recorded answers through our scorers (mygpt_eval_reference). Per metric, the
-- latest reference run that logged it (judge=false runs omit the LLM-judge metrics).
mygpt AS (
  SELECT metric, value, regexp_extract(metric, '^mygpt_ref/[^/]+/([^/]+)/', 1) AS config,
         row_number() OVER (PARTITION BY metric ORDER BY start_time DESC) AS rk
  FROM metrics
  WHERE tags['mygpt.eval.kind'] = 'reference'
)
