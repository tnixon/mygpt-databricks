-- Shared CTEs, prepended to every cost-dashboard dataset by build_cost_dashboard.py.
-- Names in braces are filled from the bundle variables (databricks.yml + variable-overrides.json)
-- and from IDs the builder resolves in the workspace (Genie space, index pipelines, app SP, ...).
-- The workspace is shared with other workloads, so every source is filtered to MyGPT's
-- resources explicitly; see dashboards/README.md for the attribution rules.
WITH
prices AS (
  SELECT sku_name, cloud, usage_unit, price_start_time,
         coalesce(price_end_time, TIMESTAMP '9999-12-31') AS price_end_time,
         CAST(coalesce(pricing.effective_list.default, pricing.default) AS DOUBLE) AS usd_per_unit
  FROM system.billing.list_prices
  WHERE currency_code = 'USD'
),
-- This workspace's billed usage, list-priced. RETRACTION / RESTATEMENT rows carry negative
-- quantities, so plain sums are correct.
usage AS (
  SELECT u.record_id, u.usage_date, u.usage_start_time, u.usage_end_time, u.sku_name, u.usage_unit,
         u.billing_origin_product AS product, u.usage_metadata AS md, u.identity_metadata AS im,
         u.product_features AS pf,
         CAST(u.usage_quantity AS DOUBLE) AS qty,
         CASE WHEN u.usage_unit = 'DBU' THEN CAST(u.usage_quantity AS DOUBLE) ELSE 0 END AS dbus,
         CAST(u.usage_quantity AS DOUBLE) * coalesce(p.usd_per_unit, 0) AS usd
  FROM system.billing.usage u
  LEFT JOIN prices p
    ON p.sku_name = u.sku_name AND p.cloud = u.cloud AND p.usage_unit = u.usage_unit
   AND u.usage_start_time >= p.price_start_time AND u.usage_start_time < p.price_end_time
  WHERE u.workspace_id = '{workspace_id}' AND u.usage_date >= date_sub(current_date(), {lookback_days})
),
-- MyGPT's jobs: the bundle's jobs (display names start with name_prefix) and the job MLflow
-- creates for the production-monitoring scorers on the shared experiment.
mygpt_jobs AS (
  SELECT job_id, name, run_as_user_name AS run_as,
         CASE WHEN name LIKE '{name_prefix} Deploy%' THEN 'deploy'
              WHEN name LIKE '{name_prefix} Eval%' OR name LIKE '%Trace Metrics Computation Job' THEN 'eval'
              ELSE 'ingest' END AS lifecycle
  FROM (SELECT * FROM system.lakeflow.jobs WHERE workspace_id = '{workspace_id}'
        QUALIFY row_number() OVER (PARTITION BY job_id ORDER BY change_time DESC) = 1)
  WHERE name LIKE '{name_prefix} %' OR name = '[{experiment_id}] Trace Metrics Computation Job'
),
-- Time windows of the eval / monitoring job runs, with their run-as identity: FMAPI judge calls
-- and AGENT_EVALUATION usage carry no resource id, only that identity.
eval_windows AS (
  SELECT r.run_id, j.run_as, min(r.period_start_time) AS t0, max(r.period_end_time) AS t1
  FROM system.lakeflow.job_run_timeline r JOIN mygpt_jobs j ON j.job_id = r.job_id
  WHERE r.workspace_id = '{workspace_id}' AND j.lifecycle = 'eval'
    AND r.period_start_time >= date_sub(current_date(), {lookback_days} + 1)
  GROUP BY r.run_id, j.run_as
),
entities AS (
  SELECT served_entity_id, endpoint_name
  FROM system.serving.served_entities
  WHERE workspace_id = '{workspace_id}'
  QUALIFY row_number() OVER (PARTITION BY served_entity_id ORDER BY change_time DESC) = 1
),
-- Every tracked serving request in the workspace (FMAPI, gateways, usage-tracked endpoints).
requests AS (
  SELECT u.databricks_request_id AS rid, e.endpoint_name, u.requester, u.request_time,
         to_date(u.request_time) AS usage_date, u.status_code,
         coalesce(u.input_token_count, 0) AS tin, coalesce(u.output_token_count, 0) AS tout,
         u.usage_context
  FROM system.serving.endpoint_usage u JOIN entities e ON e.served_entity_id = u.served_entity_id
  WHERE u.workspace_id = '{workspace_id}' AND u.request_time >= date_sub(current_date(), {lookback_days})
),
-- MyGPT's AI Gateway endpoints and what they front (agents/gateway/driver.py). The upstream
-- sees gateway traffic as the gateway token's owner, so gateway tokens are counted on the gateway.
gateways AS (
  SELECT * FROM VALUES
    ('{chat_gateway}', '{chat_upstream}'),
    ('{translate_gateway}', '{translate_upstream}'),
    ('{embeddings_gateway}', '{embedding_endpoint}')
  AS g(gateway, upstream)
),
upstreams AS (SELECT DISTINCT upstream AS endpoint_name FROM gateways),
-- The pay-per-token FMAPI endpoints MyGPT calls: shared with other workloads, so allocated.
fm_endpoints AS (SELECT DISTINCT endpoint_name FROM VALUES {fm_endpoints} AS f(endpoint_name)),
-- MyGPT's service principals: whoever calls a MyGPT-owned endpoint as an SP (the agents'
-- serving identities, Genie-calling supervisor), plus the app's SP and any extra ids.
mygpt_sps AS (
  SELECT DISTINCT requester FROM requests
  WHERE endpoint_name IN ('{chat_gateway}', '{translate_gateway}', '{embeddings_gateway}',
                          '{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}')
    AND requester RLIKE '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  UNION
  -- the RAG agent's serving identities (one per deployed version, incl. those that predate the
  -- gateways): service principals that queried a MyGPT vector index
  SELECT DISTINCT user_identity.email FROM system.access.audit
  WHERE workspace_id = '{workspace_id}' AND event_date >= date_sub(current_date(), {lookback_days} + 30)
    AND service_name = 'vectorSearch' AND action_name = 'queryVectorIndex'
    AND request_params['name'] LIKE '{catalog}.{schema}.%'
    AND user_identity.email RLIKE '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  UNION
  SELECT requester FROM VALUES {mygpt_requesters} AS x(requester)
),
eval_claimed AS (
  SELECT DISTINCT r.rid
  FROM requests r JOIN eval_windows w ON w.run_as = r.requester AND r.request_time BETWEEN w.t0 AND w.t1
),
-- FMAPI requests, each tagged with the rule that attributes it to MyGPT (NULL = someone else):
--   usage_context  the caller sent usage_context.project = mygpt (on a gateway upstream only the
--                  app's direct chat counts: the gateways' own pass-through is counted below)
--   mygpt identity the caller is one of MyGPT's service principals
--   eval window    a MyGPT eval / monitoring job's run-as identity, during that run (judges)
fm_requests AS (
  SELECT r.endpoint_name, r.usage_date, r.tin, r.tout,
         CASE WHEN r.usage_context['project'] = 'mygpt'
                   AND (up.endpoint_name IS NULL OR r.usage_context['component'] = 'app-direct') THEN 'usage_context'
              WHEN s.requester IS NOT NULL THEN 'mygpt identity'
              WHEN up.endpoint_name IS NULL AND ec.rid IS NOT NULL THEN 'eval window'
         END AS rule
  FROM requests r
  JOIN fm_endpoints f ON f.endpoint_name = r.endpoint_name
  LEFT JOIN upstreams up ON up.endpoint_name = r.endpoint_name
  LEFT JOIN mygpt_sps s ON s.requester = r.requester
  LEFT JOIN eval_claimed ec ON ec.rid = r.rid
),
-- MyGPT's tokens per FMAPI endpoint and day: claimed direct requests + gateway pass-through.
fm_mine AS (
  SELECT endpoint_name, usage_date, rule, tin, tout FROM fm_requests WHERE rule IS NOT NULL
  UNION ALL
  SELECT g.upstream, r.usage_date, 'gateway', r.tin, r.tout
  FROM requests r JOIN gateways g ON g.gateway = r.endpoint_name
  JOIN fm_endpoints f ON f.endpoint_name = g.upstream
),
-- Output tokens cost more than input ones; weight them when splitting an endpoint's bill.
fm_weights AS (
  SELECT endpoint_name, CASE WHEN endpoint_name LIKE '%claude%' THEN 5.0 ELSE 3.0 END AS w_out FROM fm_endpoints
),
fm_share AS (
  SELECT a.endpoint_name, a.usage_date, a.all_in, a.all_out,
         coalesce(m.my_in, 0) AS my_in, coalesce(m.my_out, 0) AS my_out,
         CASE WHEN a.all_in + w.w_out * a.all_out > 0
              THEN least(1.0, (coalesce(m.my_in, 0) + w.w_out * coalesce(m.my_out, 0)) / (a.all_in + w.w_out * a.all_out))
              ELSE 0.0 END AS share
  FROM (SELECT endpoint_name, usage_date, sum(tin) AS all_in, sum(tout) AS all_out FROM fm_requests GROUP BY ALL) a
  JOIN fm_weights w ON w.endpoint_name = a.endpoint_name
  LEFT JOIN (SELECT endpoint_name, usage_date, sum(tin) AS my_in, sum(tout) AS my_out FROM fm_mine GROUP BY ALL) m
    ON m.endpoint_name = a.endpoint_name AND m.usage_date = a.usage_date
),
-- Queries on the shared SQL warehouse that belong to MyGPT: its jobs, its dashboards, its Genie
-- space, and its service principals / app.
wh_queries AS (
  SELECT to_date(q.start_time) AS usage_date, q.query_source.genie_space_id AS genie_space_id,
         coalesce(q.total_task_duration_ms, 0) / 3.6e6 AS task_hours,
         (j.job_id IS NOT NULL OR q.query_source.dashboard_id IN ({dashboard_ids})
          OR q.query_source.genie_space_id = '{genie_space_id}' OR s.requester IS NOT NULL) AS is_mygpt
  FROM system.query.history q
  LEFT JOIN mygpt_jobs j ON j.job_id = q.query_source.job_info.job_id
  LEFT JOIN mygpt_sps s ON s.requester = q.executed_by
  WHERE q.workspace_id = '{workspace_id}' AND q.compute.warehouse_id = '{warehouse_id}'
    AND q.start_time >= date_sub(current_date(), {lookback_days})
),
wh_share AS (
  SELECT usage_date, sum(CASE WHEN is_mygpt THEN task_hours ELSE 0 END) AS my_hours, sum(task_hours) AS all_hours,
         count_if(is_mygpt) AS my_queries
  FROM wh_queries GROUP BY usage_date
),
-- Billed usage that belongs to MyGPT outright, with its component / category / lifecycle and
-- the native unit it is measured in. `windowed` rows count active hours from billing windows.
direct AS (
  SELECT u.*, md.endpoint_name AS component, 'Serving endpoints' AS category, 'runtime' AS lifecycle,
         'endpoint-hours' AS native_unit
  FROM usage u WHERE product = 'MODEL_SERVING'
    AND md.endpoint_name IN ('{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}')
  UNION ALL
  -- AI Gateway fees: usage tracking on the gateways, the inference tables on the agents
  SELECT u.*,
         concat(md.endpoint_name, IF(md.endpoint_name IN ('{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}'),
                                     ' (inference table)', ' (gateway)')),
         CASE WHEN md.endpoint_name IN ('{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}')
              THEN 'Serving endpoints' ELSE 'Foundation models' END, 'runtime', 'requests'
  FROM usage u WHERE product = 'AI_GATEWAY'
    AND md.endpoint_name IN ('{chat_gateway}', '{translate_gateway}', '{embeddings_gateway}',
                             '{rag_agent_endpoint}', '{supervisor_endpoint}', '{embedding_endpoint}')
  UNION ALL
  SELECT u.*, '{vs_endpoint} (endpoint)', 'Vector Search', 'runtime', 'endpoint-hours'
  FROM usage u WHERE product = 'VECTOR_SEARCH' AND md.endpoint_name = '{vs_endpoint}'
  UNION ALL
  SELECT u.*, '{vs_endpoint} (index sync)', 'Vector Search', 'ingest', 'sync-hours'
  FROM usage u WHERE product = 'VECTOR_SEARCH' AND md.dlt_pipeline_id IN ({vs_pipeline_ids})
  UNION ALL
  SELECT u.*, IF(usage_unit = 'DBU', 'Lakebase compute', 'Lakebase storage'), 'Lakebase', 'runtime',
         IF(usage_unit = 'DBU', 'compute-hours', 'DSU')
  FROM usage u WHERE product = 'LAKEBASE' AND md.project_id = '{pg_project_uid}'
  UNION ALL
  SELECT u.*, 'App {app_name}', 'App', 'runtime', 'app-hours'
  FROM usage u WHERE product = 'APPS' AND md.app_name = '{app_name}'
  UNION ALL
  SELECT u.*, 'Genie space', 'SQL & Genie', 'runtime', 'queries'
  FROM usage u WHERE product = 'GENIE' AND md.genie.agent_id = '{genie_space_id}'
  UNION ALL
  SELECT u.*, lower(coalesce(pf.ai_functions.ai_function, 'ai functions')), 'SQL & Genie', 'ingest', 'DBU'
  FROM usage u WHERE product = 'AI_FUNCTIONS' AND md.warehouse_id = '{warehouse_id}'
  UNION ALL
  SELECT u.*, j.name, 'Jobs', j.lifecycle, 'run-hours'
  FROM usage u JOIN mygpt_jobs j ON j.job_id = u.md.job_id WHERE u.product = 'JOBS'
  UNION ALL
  -- Managed MLflow judges: billed with only a run-as identity, so claimed by eval-run window.
  SELECT u.*, 'MLflow judges', 'Evaluation', 'eval', 'DBU'
  FROM usage u WHERE product = 'AGENT_EVALUATION'
    AND record_id IN (SELECT u2.record_id FROM usage u2 JOIN eval_windows w
                        ON w.run_as = u2.im.run_as AND u2.usage_start_time < w.t1 AND u2.usage_end_time > w.t0
                      WHERE u2.product = 'AGENT_EVALUATION')
),
-- Native usage that billing doesn't carry: gateway requests, job run-hours, Genie queries.
activity AS (
  SELECT usage_date, concat(endpoint_name, ' (gateway)') AS component, count(*) AS native_qty
  FROM requests
  WHERE endpoint_name IN ('{chat_gateway}', '{translate_gateway}', '{embeddings_gateway}', '{embedding_endpoint}')
  GROUP BY ALL
  UNION ALL
  -- the agents aren't usage-tracked; their inference (payload) tables log every request
  SELECT request_date, '{supervisor_endpoint} (inference table)', sum(1 / coalesce(nullif(sampling_fraction, 0), 1))
  FROM {supervisor_payload} WHERE request_date >= date_sub(current_date(), {lookback_days}) GROUP BY ALL
  UNION ALL
  SELECT request_date, '{rag_agent_endpoint} (inference table)', sum(1 / coalesce(nullif(sampling_fraction, 0), 1))
  FROM {rag_agent_payload} WHERE request_date >= date_sub(current_date(), {lookback_days}) GROUP BY ALL
  UNION ALL
  SELECT to_date(r.period_start_time), j.name,
         sum(unix_timestamp(r.period_end_time) - unix_timestamp(r.period_start_time)) / 3600.0
  FROM system.lakeflow.job_run_timeline r JOIN mygpt_jobs j ON j.job_id = r.job_id
  WHERE r.workspace_id = '{workspace_id}' AND r.period_start_time >= date_sub(current_date(), {lookback_days})
  GROUP BY ALL
  UNION ALL
  SELECT usage_date, 'Genie space', count(*) FROM wh_queries WHERE genie_space_id = '{genie_space_id}' GROUP BY ALL
),
direct_daily AS (
  -- one row per billing window first, so overlapping rows (replicas, SKUs) count the window once
  SELECT usage_date, component, category, lifecycle, native_unit,
         sum((unix_timestamp(usage_end_time) - unix_timestamp(usage_start_time)) / 3600.0) AS window_hours,
         sum(qty) AS qty, sum(dbus) AS dbus, sum(usd) AS usd
  FROM (SELECT usage_date, component, category, lifecycle, native_unit, usage_start_time, usage_end_time,
               sum(qty) AS qty, sum(dbus) AS dbus, sum(usd) AS usd
        FROM direct GROUP BY ALL)
  GROUP BY ALL
),
-- The whole MyGPT bill: one row per day x component, in native units, DBUs and list-price $.
cost_rows AS (
  SELECT d.usage_date, d.component, d.category, d.lifecycle, 'direct' AS allocation, d.native_unit,
         CASE WHEN d.native_unit IN ('endpoint-hours', 'sync-hours', 'compute-hours', 'app-hours') THEN d.window_hours
              WHEN d.native_unit IN ('requests', 'run-hours', 'queries') THEN coalesce(a.native_qty, 0)
              WHEN d.native_unit = 'DSU' THEN d.qty
              ELSE d.dbus END AS native_qty,
         d.dbus, d.usd AS list_usd
  FROM direct_daily d LEFT JOIN activity a ON a.usage_date = d.usage_date AND a.component = d.component
  UNION ALL
  -- shared FMAPI endpoints: each day's bill (serving + AI Gateway fee) x MyGPT's weighted token share
  SELECT b.usage_date, b.endpoint_name,
         'Foundation models',
         IF(b.endpoint_name = '{judge_model}', 'eval', 'runtime'),
         'token share', 'tokens',
         coalesce(s.my_in + s.my_out, 0), b.dbus * coalesce(s.share, 0), b.usd * coalesce(s.share, 0)
  FROM (SELECT usage_date, md.endpoint_name, sum(dbus) AS dbus, sum(usd) AS usd FROM usage
        WHERE product IN ('MODEL_SERVING', 'AI_GATEWAY') AND md.endpoint_name IN (SELECT endpoint_name FROM fm_endpoints)
        GROUP BY ALL) b
  LEFT JOIN fm_share s ON s.endpoint_name = b.endpoint_name AND s.usage_date = b.usage_date
  UNION ALL
  -- shared SQL warehouse: each day's bill x MyGPT's share of its query task time
  SELECT b.usage_date, 'SQL warehouse {warehouse_id}', 'SQL & Genie', 'runtime', 'query share', 'query-hours',
         coalesce(s.my_hours, 0),
         b.dbus * coalesce(s.my_hours / nullif(s.all_hours, 0), 0), b.usd * coalesce(s.my_hours / nullif(s.all_hours, 0), 0)
  FROM (SELECT usage_date, sum(dbus) AS dbus, sum(usd) AS usd FROM usage
        WHERE product = 'SQL' AND md.warehouse_id = '{warehouse_id}' GROUP BY ALL) b
  LEFT JOIN wh_share s ON s.usage_date = b.usage_date
)
