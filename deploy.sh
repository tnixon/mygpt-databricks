#!/usr/bin/env bash
# MyGPT deployment: the DEPLOYMENT.md steps as plain databricks CLI commands.
# Run it top to bottom, or copy-paste one command at a time (run from the repo root).
#
# Before starting:
#   - set targets.default.workspace.host in databricks.yml and app/databricks.yml to your workspace
#   - set warehouse_id in .databricks/bundle/default/variable-overrides.json
#     and in app/.databricks/bundle/default/variable-overrides.json
#   - the catalog must already exist
#   - point EVAL_DIR below at your local copy of St. Jude's MyGPT-evaluations repo
set -e

# Must match catalog / schema / docs_volume / reference_volume / secret_scope in variable-overrides.json.
PROFILE=mygpt
CATALOG=structural_biology
SCHEMA=mygpt
DOCS=dbfs:/Volumes/$CATALOG/$SCHEMA/docs
REF=dbfs:/Volumes/$CATALOG/$SCHEMA/reference
SCOPE=mygpt
# Local copy of the MyGPT-evaluations repo (any folder): corpus PDFs + reference CSVs.
EVAL_DIR=reference/MyGPT-evaluations

# 1. Schema and the experiment's parent folder
databricks schemas create $SCHEMA $CATALOG -p $PROFILE
databricks workspace mkdirs /Shared/mygpt -p $PROFILE

# 2. Platform bundle: volumes, secret scope, Vector Search endpoint, Lakebase, experiment, dashboard, jobs
databricks bundle deploy -p $PROFILE

# 3. bge-m3 embedding endpoint
databricks bundle run mygpt_deploy_embedding -p $PROFILE

# 4. Gateway upstream token (paste a service-principal token when prompted)
databricks secrets put-secret $SCOPE gw_upstream_token -p $PROFILE

# 5. AI Gateway endpoints
databricks bundle run mygpt_deploy_gateway -p $PROFILE
databricks bundle run mygpt_deploy_chat_gateway -p $PROFILE
databricks bundle run mygpt_deploy_translate_gateway -p $PROFILE

# 6. Upload the corpora and the reference CSVs (Genie seed tables, eval datasets) from EVAL_DIR
test -f "$EVAL_DIR/kinase_literature/inputs/human_kinome_synonyms.csv" \
  || { echo "EVAL_DIR=$EVAL_DIR is not a MyGPT-evaluations folder; set EVAL_DIR at the top of deploy.sh" >&2; exit 1; }
databricks fs cp -r --overwrite "$EVAL_DIR/health_policies/inputs/French_10_pdfs" $DOCS/health_policies_fr -p $PROFILE
databricks fs cp -r --overwrite "$EVAL_DIR/health_policies/inputs/Spanish_10_pdfs" $DOCS/health_policies_es -p $PROFILE
databricks fs cp -r --overwrite "$EVAL_DIR/BioASQ/inputs/pdfs" $DOCS/biomed -p $PROFILE
# Skips CSVs with ':' in the name (the CLI can't copy them; no job reads them).
(cd "$EVAL_DIR" && find . -name '*.csv' ! -name '*:*') | while IFS= read -r f; do
  f=${f#./}
  databricks fs mkdir "$REF/MyGPT-evaluations/$(dirname "$f")" -p $PROFILE   # fs cp of a file won't create it
  databricks fs cp --overwrite "$EVAL_DIR/$f" "$REF/MyGPT-evaluations/$f" -p $PROFILE
done
# optional: databricks fs cp -r --overwrite <dir of kinase PDFs> $DOCS/kinase_literature -p $PROFILE

# 7. ETL (also creates the Vector Search index on the first run)
databricks bundle run mygpt_etl -p $PROFILE

# 8. RAG agent
databricks bundle run mygpt_deploy_agent -p $PROFILE

# 9. Tools, Genie space, Supervisor
databricks bundle run mygpt_deploy_tools -p $PROFILE
databricks bundle run mygpt_deploy_genie -p $PROFILE
databricks bundle run mygpt_deploy_supervisor -p $PROFILE

# 10. App. It binds the platform bundle's mygpt_upload_ingest job and experiment by id; look them
#     up from the platform bundle and pass them as --var (overrides variable-overrides.json).
SUMMARY=$(databricks bundle summary -p $PROFILE -o json)
UPLOAD_JOB_ID=$(printf %s "$SUMMARY" | python3 -c "import json,sys; print(json.load(sys.stdin)['resources']['jobs']['mygpt_upload_ingest']['id'])")
EXPERIMENT_ID=$(printf %s "$SUMMARY" | python3 -c "import json,sys; print(json.load(sys.stdin)['resources']['experiments']['mygpt_rag_agent']['id'])")
echo "upload_job_id=$UPLOAD_JOB_ID experiment_id=$EXPERIMENT_ID"
cd app
databricks bundle deploy -p $PROFILE --var upload_job_id=$UPLOAD_JOB_ID --var experiment_id=$EXPERIMENT_ID
databricks bundle run app -p $PROFILE --var upload_job_id=$UPLOAD_JOB_ID --var experiment_id=$EXPERIMENT_ID
cd ..

# 11. Optional: evaluation and monitoring
# databricks fs cp -r --overwrite "reference/MyGPT-evaluations-private-main/ground truths" dbfs:/Volumes/$CATALOG/$SCHEMA/eval_private -p $PROFILE
# databricks bundle run mygpt_eval_build_datasets -p $PROFILE
# databricks bundle run mygpt_eval -p $PROFILE
# databricks bundle run mygpt_eval_reference -p $PROFILE
# databricks bundle run mygpt_eval_monitor -p $PROFILE
