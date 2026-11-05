-- DBUs per hour of the custom serving endpoints (= their burn rate; flat lines = always-on).
SELECT date_trunc('HOUR', usage_start_time) AS hour, usage_date, md.endpoint_name AS endpoint, sum(dbus) AS dbus, sum(usd) AS list_usd
FROM usage
WHERE product = 'MODEL_SERVING'
  AND md.endpoint_name IN ('{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}')
GROUP BY ALL
ORDER BY hour
