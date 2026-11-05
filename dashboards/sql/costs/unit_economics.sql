-- Daily cost next to the traffic it served: supervisor requests = grounded answers in the app.
SELECT c.usage_date, c.list_usd, c.runtime_usd, coalesce(s.answers, 0) AS answers, coalesce(r.rag_requests, 0) AS rag_requests
FROM (SELECT usage_date, sum(list_usd) AS list_usd, sum(IF(lifecycle = 'runtime', list_usd, 0)) AS runtime_usd
      FROM cost_rows GROUP BY usage_date) c
LEFT JOIN (SELECT request_date AS usage_date, sum(1 / coalesce(nullif(sampling_fraction, 0), 1)) AS answers
           FROM {supervisor_payload} WHERE status_code = 200 GROUP BY 1) s ON s.usage_date = c.usage_date
LEFT JOIN (SELECT request_date AS usage_date, sum(1 / coalesce(nullif(sampling_fraction, 0), 1)) AS rag_requests
           FROM {rag_agent_payload} WHERE status_code = 200 GROUP BY 1) r ON r.usage_date = c.usage_date
ORDER BY c.usage_date
