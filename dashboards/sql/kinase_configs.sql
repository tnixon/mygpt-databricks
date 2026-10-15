-- BlueSky per question: our agent vs each MyGPT configuration (same scorers), plus SME accuracy.
SELECT CASE b.ours_metric WHEN 'expression_system_correct/mean' THEN 'Q11 expression system'
                          WHEN 'figure_ref_match/mean' THEN 'Q17 figure references'
                          ELSE 'Q16v2 assay (LLM judge)' END AS question,
       concat('MyGPT ', m.config) AS system, round(m.value, 3) AS score
FROM benchmarks b JOIN mygpt m ON m.rk = 1 AND m.metric RLIKE b.mygpt_rx
WHERE b.use_case = 'bluesky'
UNION ALL
SELECT CASE b.ours_metric WHEN 'expression_system_correct/mean' THEN 'Q11 expression system'
                          WHEN 'figure_ref_match/mean' THEN 'Q17 figure references'
                          ELSE 'Q16v2 assay (LLM judge)' END,
       concat('Databricks agent (', o.model_alias, ')'), round(o.value, 3)
FROM benchmarks b JOIN ours o ON o.rk = 1 AND o.use_case = b.use_case AND o.metric = b.ours_metric
WHERE b.use_case = 'bluesky'
