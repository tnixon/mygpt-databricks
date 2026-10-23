# MyGPT on Databricks — POC monorepo
# Common tasks. Run `make help` to list targets.
# Override the target workspace with PROFILE=<name> (default: mygpt).
# Bundle commands read names / ids from the bundles' variable-overrides.json (`make config`).
# The app standardizes on pnpm — see app/package.json `packageManager`
# (pnpm@11.0.8, provisioned via corepack) + app/pnpm-workspace.yaml. Don't run
# `npm install` in app/ (it would re-create package-lock.json and diverge from
# the pnpm-lock.yaml the Databricks Apps build compute installs from).

PROFILE ?= mygpt
APP_DIR := app

# Used by the non-bundle commands below; must match catalog / schema / secret_scope / app_name
# in variable-overrides.json (override on the command line, e.g. make eval-corpus CATALOG=x).
CATALOG ?= structural_biology
SCHEMA ?= mygpt
SCOPE ?= mygpt
APP_NAME ?= st-jude-mygpt

# Phase-2 eval corpora: one docs-volume folder per corpus (= the agent's custom_inputs.corpus).
DOCS_VOLUME := dbfs:/Volumes/$(CATALOG)/$(SCHEMA)/docs
# Local copy of St. Jude's MyGPT-evaluations repo (any folder; EVAL_INPUTS=<dir>).
EVAL_INPUTS ?= reference/MyGPT-evaluations
# Its CSVs (Genie seed tables, eval datasets), at their repo paths; must match reference_volume.
REFERENCE_VOLUME := dbfs:/Volumes/$(CATALOG)/$(SCHEMA)/reference
# St. Jude's private ground truths (gitignored) -> the restricted eval_private UC volume.
EVAL_PRIVATE := reference/MyGPT-evaluations-private-main/ground truths
PRIVATE_VOLUME := dbfs:/Volumes/$(CATALOG)/$(SCHEMA)/eval_private
KINASE_PDFS ?=   # dir of St. Jude's kinase PDFs, named <pubmed_id>.pdf (not in the submodule)

.DEFAULT_GOAL := help

.PHONY: help init config submodules submodules-update install dev build typecheck lint test \
	validate deploy deploy-platform deploy-app app-run secrets gateways agents summary app-logs \
	clean eval-corpus reference-csvs corpus-hi eval-private eval-test
# Job chains (gateways, agents) must run in order, and deploy-platform must finish first.
.NOTPARALLEL:

help: ## List available targets
	@grep -E '^[a-zA-Z_%-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: submodules install ## First-time setup: fetch submodules + install app deps

# The bundle CLI reads .databricks/bundle/<target>/variable-overrides.json (gitignored, per checkout).
config: ## Install both bundles' variable-overrides.json from the committed examples (never overwrites)
	@for d in . $(APP_DIR); do \
	  f=$$d/.databricks/bundle/default/variable-overrides.json; \
	  if [ -f $$f ]; then echo "kept      $$f"; \
	  else mkdir -p $$(dirname $$f) && cp $$d/variable-overrides.example.json $$f && echo "installed $$f"; fi; \
	done

submodules: ## Fetch/init reference submodules (stjude/MyGPT, MyGPT-evaluations)
	git submodule update --init --recursive

submodules-update: ## Bump reference submodules to latest upstream (then commit the pointers)
	git submodule update --remote --recursive
	@echo "Submodules bumped — review and commit the updated pointers."

install: ## Install app dependencies (pnpm; corepack auto-selects the pinned pnpm)
	cd $(APP_DIR) && corepack enable && pnpm install

dev: ## Run the app locally (http://localhost:8000)
	cd $(APP_DIR) && pnpm dev

build: ## Build the app
	cd $(APP_DIR) && pnpm build

typecheck: ## Type-check the app
	cd $(APP_DIR) && pnpm typecheck

lint: ## Lint the app
	cd $(APP_DIR) && pnpm lint

test: ## Run app tests
	cd $(APP_DIR) && pnpm test

# ---------------------------------------------------------------- deployment (DEPLOYMENT.md)
# Both bundles: platform (repo root) owns the jobs / volumes / Lakebase, app/ owns the App.
# deploy.sh is the same sequence as plain commands.
validate: ## Validate both bundles (PROFILE=<name>)
	databricks bundle validate -p $(PROFILE)
	cd $(APP_DIR) && databricks bundle validate -p $(PROFILE)

deploy: deploy-platform deploy-app ## Deploy both bundles (PROFILE=<name>)

deploy-platform: ## Deploy the platform bundle: volumes, secret scope, VS endpoint, Lakebase, experiment, dashboard, jobs (step 2)
	databricks bundle deploy -p $(PROFILE)

deploy-app: ## Deploy the app bundle (set upload_job_id / experiment_id in its overrides first; see `make summary`)
	cd $(APP_DIR) && databricks bundle deploy -p $(PROFILE)

app-run: deploy-app ## Deploy the app bundle, then (re)start the app on the new code (step 12)
	cd $(APP_DIR) && databricks bundle run app -p $(PROFILE)

# `bundle run` executes the last-DEPLOYED code, so every run deploys the platform bundle first
# (once per make invocation): make run-etl, make run-deploy_agent, make run-eval_monitor, ...
run-%: deploy-platform ## Deploy the platform bundle, then run job mygpt_<%> (e.g. run-etl, run-deploy_agent)
	databricks bundle run mygpt_$* -p $(PROFILE)

gateways: run-deploy_gateway run-deploy_chat_gateway run-deploy_translate_gateway ## Deploy the 3 AI Gateway endpoints (step 5; needs `secrets`)

agents: run-deploy_tools run-deploy_genie run-deploy_supervisor ## Deploy the tools, Genie space and Supervisor, in order (step 11)

secrets: ## Store the gateway upstream token, prompted (step 4; AZURE=1 also stores the Azure AI Foundry key)
	databricks secrets put-secret $(SCOPE) gw_upstream_token -p $(PROFILE)
	@if [ -n "$(AZURE)" ]; then databricks secrets put-secret $(SCOPE) azure_foundry_key -p $(PROFILE); fi

summary: ## Show both bundles' deployed resources (ids, URLs)
	databricks bundle summary -p $(PROFILE)
	cd $(APP_DIR) && databricks bundle summary -p $(PROFILE)

app-logs: ## Show the app's recent logs (APP_LINES=<n>, default 200; FOLLOW=1 keeps streaming)
	databricks apps logs $(APP_NAME) --tail-lines $(or $(APP_LINES),200) $(if $(FOLLOW),-f) -p $(PROFILE)

clean: ## Remove app build artifacts + node_modules
	cd $(APP_DIR) && pnpm run clean

eval-corpus: reference-csvs ## Upload the eval corpora + reference CSVs (PROFILE=<name>, EVAL_INPUTS=<dir>, KINASE_PDFS=<dir>)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/health_policies/inputs/French_10_pdfs" $(DOCS_VOLUME)/health_policies_fr --profile $(PROFILE)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/health_policies/inputs/Spanish_10_pdfs" $(DOCS_VOLUME)/health_policies_es --profile $(PROFILE)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/BioASQ/inputs/pdfs" $(DOCS_VOLUME)/biomed --profile $(PROFILE)
	@if [ -n "$(KINASE_PDFS)" ]; then \
	  databricks fs cp -r --overwrite "$(KINASE_PDFS)" $(DOCS_VOLUME)/kinase_literature --profile $(PROFILE); \
	else echo "KINASE_PDFS not set — skipping kinase_literature (St. Jude to provide the PDFs)"; fi

reference-csvs: ## Upload MyGPT-evaluations' CSVs to the reference volume (step 6; EVAL_INPUTS=<dir>)
	@test -f "$(EVAL_INPUTS)/kinase_literature/inputs/human_kinome_synonyms.csv" || (echo "EVAL_INPUTS=$(EVAL_INPUTS) is not a MyGPT-evaluations folder"; exit 1)
	@# Skips CSVs with ':' in the name (the CLI can't copy them; no job reads them).
	cd "$(EVAL_INPUTS)" && find . -name '*.csv' ! -name '*:*' | while IFS= read -r f; do \
	  f=$${f#./}; \
	  databricks fs mkdir "$(REFERENCE_VOLUME)/MyGPT-evaluations/$$(dirname "$$f")" --profile $(PROFILE) && \
	  databricks fs cp --overwrite "$$f" "$(REFERENCE_VOLUME)/MyGPT-evaluations/$$f" --profile $(PROFILE) || exit 1; \
	done

corpus-hi: ## Upload the Phase-6 public Hindi health-policy PDFs to the docs volume (PROFILE=<name>, HI_PDFS=<dir>)
	@test -n "$(HI_PDFS)" || (echo "set HI_PDFS=<dir of public Hindi PDFs> (see etl/README.md, health_policies_hi)"; exit 1)
	databricks fs cp -r --overwrite "$(HI_PDFS)" $(DOCS_VOLUME)/health_policies_hi --profile $(PROFILE)

eval-private: ## Upload St. Jude's private ground truths to the restricted eval_private volume (PROFILE=<name>)
	@test -d "$(EVAL_PRIVATE)" || (echo "missing $(EVAL_PRIVATE) (St. Jude's private ground truths)"; exit 1)
	databricks fs cp -r --overwrite "$(EVAL_PRIVATE)" $(PRIVATE_VOLUME) --profile $(PROFILE)

eval-test: ## Unit-test the eval harness, agent scoring / provided-sources / multilingual, Supervisor + tools, ETL parse fallback (no workspace needed)
	uv run --no-project --with-requirements eval/requirements.txt --with PyStemmer --with pymupdf pytest -q eval/ agents/rag_agent/test_scoring.py agents/rag_agent/test_provided_sources.py agents/rag_agent/test_multilingual.py agents/supervisor/test_supervisor.py agents/tools/test_tools.py etl/test_parse_fallback.py
