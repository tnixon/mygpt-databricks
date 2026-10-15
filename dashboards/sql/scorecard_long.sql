-- Scorecard in long form (one row per line x system) for the grouped ours-vs-MyGPT bar.
SELECT b.sort, concat(o.dataset_label, ' | ', b.benchmark) AS line, o.model_alias,
       'Databricks agent' AS system, round(o.value, 3) AS score
FROM benchmarks b JOIN ours o ON o.rk = 1 AND o.use_case = b.use_case AND o.metric = b.ours_metric
WHERE b.mygpt_rx IS NOT NULL
UNION ALL
SELECT b.sort, concat(o.dataset_label, ' | ', b.benchmark), o.model_alias,
       'MyGPT (best config)', round(max(m.value), 3)
FROM benchmarks b
JOIN ours o ON o.rk = 1 AND o.use_case = b.use_case AND o.metric = b.ours_metric
JOIN mygpt m ON m.rk = 1 AND m.metric RLIKE b.mygpt_rx
GROUP BY b.sort, o.dataset_label, b.benchmark, o.model_alias
