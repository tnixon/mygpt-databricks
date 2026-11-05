-- One row per MyGPT job run: duration, outcome, DBUs and list $.
SELECT j.name AS job, j.lifecycle, r.run_id, r.started, to_date(r.started) AS usage_date, r.run_hours, r.result_state,
       coalesce(b.dbus, 0) AS dbus, coalesce(b.usd, 0) AS list_usd
FROM (SELECT job_id, run_id, min(period_start_time) AS started,
             sum(unix_timestamp(period_end_time) - unix_timestamp(period_start_time)) / 3600.0 AS run_hours,
             max_by(result_state, period_end_time) AS result_state
      FROM system.lakeflow.job_run_timeline
      WHERE workspace_id = '{workspace_id}' AND period_start_time >= date_sub(current_date(), {lookback_days})
      GROUP BY job_id, run_id) r
JOIN mygpt_jobs j ON j.job_id = r.job_id
LEFT JOIN (SELECT md.job_run_id AS run_id, sum(dbus) AS dbus, sum(usd) AS usd FROM usage WHERE product = 'JOBS' GROUP BY 1) b
  ON b.run_id = r.run_id
ORDER BY r.started DESC
