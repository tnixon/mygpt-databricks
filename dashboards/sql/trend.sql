-- Headline metrics of every full eval run, for the regression trend across agent versions.
SELECT o.start_time AS run_time, date_format(o.start_time, 'yyyy-MM-dd HH:mm') AS run_label,
       o.dataset_label, o.model_alias, o.agent_version, o.run_id,
       l.metric_label, concat(o.dataset_label, ' | ', l.metric_label) AS series,
       round(o.value, 3) AS score
FROM ours o
JOIN LATERAL (SELECT coalesce(b.benchmark, CASE o.metric
         WHEN 'retrieval_groundedness/mean' THEN 'Groundedness (judge)'
         WHEN 'relevance_to_query/mean' THEN 'Relevance to query (judge)'
         WHEN 'expectations_guidelines/mean' THEN 'Per-row guidelines (judge)'
         WHEN 'mygpt_grounding/mean' THEN 'MyGPT grounding (judge)' END) AS metric_label) l LEFT JOIN benchmarks b ON b.use_case = o.use_case AND b.ours_metric = o.metric
WHERE b.sort IS NOT NULL
   OR o.metric IN ('retrieval_groundedness/mean', 'relevance_to_query/mean',
                   'expectations_guidelines/mean', 'mygpt_grounding/mean')
