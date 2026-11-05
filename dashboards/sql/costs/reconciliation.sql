-- The workspace's whole bill split into MyGPT (everything attributed above) and other workloads.
SELECT t.usage_date, 'MyGPT' AS owner, coalesce(m.usd, 0) AS list_usd
FROM (SELECT usage_date, sum(usd) AS usd FROM usage GROUP BY 1) t
LEFT JOIN (SELECT usage_date, sum(list_usd) AS usd FROM cost_rows GROUP BY 1) m ON m.usage_date = t.usage_date
UNION ALL
SELECT t.usage_date, 'Other workloads', t.usd - coalesce(m.usd, 0)
FROM (SELECT usage_date, sum(usd) AS usd FROM usage GROUP BY 1) t
LEFT JOIN (SELECT usage_date, sum(list_usd) AS usd FROM cost_rows GROUP BY 1) m ON m.usage_date = t.usage_date
ORDER BY usage_date
