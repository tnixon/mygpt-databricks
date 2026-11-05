-- Requests per endpoint and day: the agents from their inference (payload) tables, the gateways and
-- usage-tracked endpoints from system.serving.endpoint_usage.
SELECT request_date AS usage_date, '{supervisor_endpoint}' AS endpoint,
       sum(1 / coalesce(nullif(sampling_fraction, 0), 1)) AS requests, count_if(status_code >= 400) AS errors
FROM {supervisor_payload}
WHERE request_date >= date_sub(current_date(), {lookback_days})
GROUP BY ALL
UNION ALL
SELECT request_date, '{rag_agent_endpoint}', sum(1 / coalesce(nullif(sampling_fraction, 0), 1)), count_if(status_code >= 400)
FROM {rag_agent_payload}
WHERE request_date >= date_sub(current_date(), {lookback_days})
GROUP BY ALL
UNION ALL
SELECT usage_date, endpoint_name, count(*), count_if(status_code >= 400)
FROM requests
WHERE endpoint_name IN ('{chat_gateway}', '{translate_gateway}', '{embeddings_gateway}', '{embedding_endpoint}')
GROUP BY ALL
ORDER BY usage_date
