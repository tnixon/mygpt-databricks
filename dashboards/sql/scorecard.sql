-- One row per benchmark line x dataset (FR / ES are separate rows) x model alias.
-- :tolerance_pct (dashboard parameter): |% difference| below it counts as on par.
, mygpt_by_benchmark AS (
  SELECT b.sort, min(m.value) AS mygpt_min, max(m.value) AS mygpt_best, count(*) AS n_configs,
         array_join(array_sort(collect_list(concat(m.config, ' ', format_number(m.value, 3)))), ', ') AS mygpt_configs
  FROM benchmarks b JOIN mygpt m ON m.rk = 1 AND m.metric RLIKE b.mygpt_rx
  GROUP BY b.sort
),
lines AS (
  SELECT b.sort, b.use_case_label, b.benchmark, b.mygpt_basis, o.dataset_label, o.model_alias,
         o.agent_version, o.start_time AS run_time, o.run_id, o.value AS ours,
         mb.mygpt_best, mb.mygpt_min, mb.n_configs, mb.mygpt_configs,
         CAST(o.params[concat('gate/', b.ours_metric)] AS DOUBLE) AS gate
  FROM benchmarks b
  JOIN ours o ON o.rk = 1 AND o.use_case = b.use_case AND o.metric = b.ours_metric
  LEFT JOIN mygpt_by_benchmark mb ON mb.sort = b.sort
)
SELECT sort, use_case_label, dataset_label, benchmark, model_alias, agent_version, run_time, run_id,
       round(ours, 3) AS ours,
       round(mygpt_best, 3) AS mygpt_best,
       CASE WHEN n_configs > 1 THEN concat(format_number(mygpt_min, 2), ' - ', format_number(mygpt_best, 2))
            ELSE format_number(mygpt_best, 3) END AS mygpt_range,
       mygpt_configs, mygpt_basis,
       round(ours - mygpt_best, 3) AS delta,
       CASE WHEN ours - mygpt_best > 0 THEN '+' ELSE '' END AS delta_sign,
       round(pct_diff, 1) AS pct_diff,
       CASE WHEN mygpt_best IS NULL THEN '-'
            WHEN abs(pct_diff) < :tolerance_pct THEN 'On par'
            WHEN pct_diff > 0 THEN 'Better'
            ELSE 'Worse' END AS status,
       CAST(mygpt_best IS NOT NULL AND pct_diff > -:tolerance_pct AS INT) AS is_at_or_above,
       CAST(mygpt_best IS NOT NULL AND pct_diff <= -:tolerance_pct AS INT) AS is_below,
       gate,
       CASE WHEN gate IS NULL THEN '-' WHEN ours >= gate THEN 'Pass' ELSE 'Fail' END AS gate_status,
       CAST(gate IS NOT NULL AND ours >= gate AS INT) AS gate_passed,
       CAST(gate IS NOT NULL AS INT) AS gated,
       concat(dataset_label, ' | ', benchmark) AS line
FROM (SELECT *, 100 * (ours - mygpt_best) / nullif(mygpt_best, 0) AS pct_diff FROM lines)
ORDER BY sort, dataset_label
