#!/usr/bin/env bash
# MyGPT deployment: the DEPLOYMENT.md steps as plain databricks CLI commands.
# Run it top to bottom, or copy-paste one command at a time (run from the repo root).
#
# Before starting:
#   - set targets.default.workspace.host in databricks.yml and app/databricks.yml to your workspace
#   - set warehouse_id in .databricks/bundle/default/variable-overrides.json
#     and in app/.databricks/bundle/default/variable-overrides.json
#   - the catalog must already exist
set -e

# Must match catalog / schema / docs_volume / secret_scope in variable-overrides.json.
PROFILE=mygpt
CATALOG=structural_biology
SCHEMA=mygpt
DOCS=dbfs:/Volumes/$CATALOG/$SCHEMA/docs
SCOPE=mygpt

# 1. Schema and the experiment's parent folder
databricks schemas create $SCHEMA $CATALOG -p $PROFILE
databricks workspace mkdirs /Shared/mygpt -p $PROFILE

# 2. Platform bundle: volumes, secret scope, Vector Search endpoint, Lakebase, experiment, dashboard, jobs
databricks bundle deploy -p $PROFILE

# 3. bge-m3 embedding endpoint
databricks bundle run mygpt_deploy_embedding -p $PROFILE

# 4. Gateway upstream token (paste a service-principal token when prompted)
databricks secrets put-secret $SCOPE gw_upstream_token -p $PROFILE
# optional: the LangSmith API key turns on the agents' LangSmith trace mirror (steps 8, 9)
# databricks secrets put-secret $SCOPE langsmith_api_key -p $PROFILE

# 5. AI Gateway endpoints
databricks bundle run mygpt_deploy_gateway -p $PROFILE
databricks bundle run mygpt_deploy_chat_gateway -p $PROFILE
databricks bundle run mygpt_deploy_translate_gateway -p $PROFILE

# 6. Upload the corpora (needs: git submodule update --init --recursive)
databricks fs cp -r --overwrite reference/MyGPT-evaluations/health_policies/inputs/French_10_pdfs $DOCS/health_policies_fr -p $PROFILE
databricks fs cp -r --overwrite reference/MyGPT-evaluations/health_policies/inputs/Spanish_10_pdfs $DOCS/health_policies_es -p $PROFILE
databricks fs cp -r --overwrite reference/MyGPT-evaluations/BioASQ/inputs/pdfs $DOCS/biomed -p $PROFILE
# optional: databricks fs cp -r --overwrite <dir of kinase PDFs> $DOCS/kinase_literature -p $PROFILE

# 7. ETL (also creates the Vector Search index on the first run)
databricks bundle run mygpt_etl -p $PROFILE

# 8. RAG agent
databricks bundle run mygpt_deploy_agent -p $PROFILE

# 9. Tools, Genie space, Supervisor
databricks bundle run mygpt_deploy_tools -p $PROFILE
databricks bundle run mygpt_deploy_genie -p $PROFILE
databricks bundle run mygpt_deploy_supervisor -p $PROFILE

# 10. App. First copy the mygpt_upload_ingest job id and the experiment id from this output
#     into upload_job_id / experiment_id in app/.databricks/bundle/default/variable-overrides.json:
databricks bundle summary -p $PROFILE
read -p "Set upload_job_id and experiment_id in the app's variable-overrides.json, then press Enter "
cd app
databricks bundle deploy -p $PROFILE
databricks bundle run app -p $PROFILE
cd ..

# 11. Optional: evaluation and monitoring
# databricks fs cp -r --overwrite "reference/MyGPT-evaluations-private-main/ground truths" dbfs:/Volumes/$CATALOG/$SCHEMA/eval_private -p $PROFILE
# databricks bundle run mygpt_eval_build_datasets -p $PROFILE
# databricks bundle run mygpt_eval -p $PROFILE
# databricks bundle run mygpt_eval_reference -p $PROFILE
# databricks bundle run mygpt_eval_monitor -p $PROFILE
