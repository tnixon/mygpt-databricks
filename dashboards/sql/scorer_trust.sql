-- How often each gold scorer agrees with St. Jude's SMEs, on MyGPT's kinase answers.
, agreement AS (
  SELECT regexp_extract(metric, '^mygpt_ref/bluesky/([^/]+)/', 1) AS config,
         coalesce(nullif(regexp_extract(metric, '/(sme|sme_realigned)/', 1), ''), 'all') AS gold_slice,
         regexp_extract(metric, '/([a-z_]+)/agreement$', 1) AS scorer,
         value AS agreement,
         regexp_replace(metric, '/agreement$', '/n') AS n_metric
  FROM mygpt WHERE rk = 1 AND metric RLIKE '^mygpt_ref/bluesky/.*/agreement$'
)
SELECT a.config, a.gold_slice, a.scorer,
       CASE a.scorer WHEN 'expression_system_correct' THEN 'Q11 expression system'
                     WHEN 'figure_ref_match' THEN 'Q17 figure references'
                     WHEN 'gold_correctness' THEN 'Q16v2 assay (LLM judge)' END AS question,
       concat(CASE a.scorer WHEN 'expression_system_correct' THEN 'Q11 expression'
                            WHEN 'figure_ref_match' THEN 'Q17 figures' ELSE 'Q16v2 assay (judge)' END,
              ' (', a.gold_slice, ')') AS label,
       round(a.agreement, 3) AS agreement, CAST(n.value AS INT) AS n
FROM agreement a LEFT JOIN mygpt n ON n.rk = 1 AND n.metric = a.n_metric
ORDER BY question, a.config, a.gold_slice
