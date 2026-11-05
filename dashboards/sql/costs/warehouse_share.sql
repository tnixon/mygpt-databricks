-- Query task-hours on the shared SQL warehouse: MyGPT's vs everyone else's (drives its cost share).
SELECT usage_date, IF(is_mygpt, 'MyGPT', 'Other workloads') AS owner, sum(task_hours) AS query_hours, count(*) AS queries
FROM wh_queries
GROUP BY ALL
ORDER BY usage_date
