# TODO

## LangSmith tracing: turn it on (code done on branch `higher_model_evals`, not yet deployed)

The RAG agent and the Supervisor now mirror their MLflow spans to LangSmith. This stays off until
the API-key secret exists. Design and settings are in DEPLOYMENT.md step 4 and agents/README.md.

### Before enabling
- [ ] Confirm St. Jude approves sending user questions, document chunks and answers to LangSmith.
      If not, set `langsmith_hide_content: "true"`. That sends only the run tree, timings and
      labels (route, corpus, model, ids).
- [ ] Check that Model Serving can reach the LangSmith API (`langsmith_endpoint`). If the
      workspace restricts outbound traffic, allow that host.

### Enable
- [ ] `cp .env.example .env` and fill in `LANGSMITH_API_KEY` (plus `LANGSMITH_WORKSPACE_ID` if the
      key is org-scoped). `.env` is gitignored and never synced.
- [ ] In `.databricks/bundle/default/variable-overrides.json`, set `langsmith_project`,
      `langsmith_endpoint` and `langsmith_workspace_id` as needed. The defaults are `mygpt`,
      `https://api.smith.langchain.com` and `""`.
- [ ] `make langsmith-secret`: stores `mygpt/langsmith_api_key`, read from `.env` over stdin.
- [ ] `make run-deploy_agent run-deploy_supervisor`. The job logs should say
      `LangSmith tracing on (project mygpt)`.

### Verify
- [ ] `python agents/supervisor/smoke_test.py`. LangSmith should show one `mygpt-supervisor` tree,
      with `mygpt-rag-agent` nested under its `retrieve` run.
- [ ] MLflow traces in `/Shared/mygpt/mygpt-rag-agent` are unchanged.
- [ ] Responses carry `custom_outputs.langsmith_run_id`.
- [ ] (If redacting) set `langsmith_hide_content: "true"`, redeploy both agents, and confirm the
      runs show no inputs or outputs.

### Notes
- To turn it off: delete the secret and redeploy both agents. Any setting change also needs a
  redeploy, because the settings become endpoint env vars.
- Tests: `make eval-test` (121 passing when this was written).
