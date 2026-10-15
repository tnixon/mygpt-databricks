-- Run list (full eval runs + reference runs) with links into the MLflow experiment.
SELECT start_time AS run_time, run_name,
       coalesce(tags['mygpt.eval.kind'], 'eval') AS kind,
       tags['mygpt.eval.model_alias'] AS model_alias,
       tags['mygpt.eval.served_model'] AS served_model,
       tags['mygpt.eval.max_rows'] AS max_rows,
       run_id,
       concat(tags['mlflow.databricks.workspaceURL'], '/ml/experiments/', experiment_id, '/runs/', run_id) AS run_url
FROM runs
WHERE tags['mygpt.eval.use_case'] IS NOT NULL OR tags['mygpt.eval.kind'] = 'reference'
ORDER BY start_time DESC
