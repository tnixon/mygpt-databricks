-- FMAPI tokens per model and day, by who they are attributed to (MyGPT's rules vs other workloads).
SELECT usage_date, model, token_type, attribution, IF(attribution = 'other workloads', 'Other workloads', 'MyGPT') AS owner, tokens
FROM (
  SELECT usage_date, endpoint_name AS model, 'input' AS token_type, rule AS attribution, sum(tin) AS tokens FROM fm_mine GROUP BY ALL
  UNION ALL
  SELECT usage_date, endpoint_name, 'output', rule, sum(tout) FROM fm_mine GROUP BY ALL
  UNION ALL
  SELECT usage_date, endpoint_name, 'input', 'other workloads', greatest(all_in - my_in, 0) FROM fm_share
  UNION ALL
  SELECT usage_date, endpoint_name, 'output', 'other workloads', greatest(all_out - my_out, 0) FROM fm_share
)
WHERE tokens > 0
ORDER BY usage_date
