# MyGPT on Databricks — POC monorepo
# Common tasks. Run `make help` to list targets.
# Override the target workspace with PROFILE=<name> (default: stjude-mygpt).
# The app standardizes on pnpm — see app/package.json `packageManager`
# (pnpm@11.0.8, provisioned via corepack) + app/pnpm-workspace.yaml. Don't run
# `npm install` in app/ (it would re-create package-lock.json and diverge from
# the pnpm-lock.yaml the Databricks Apps build compute installs from).

PROFILE ?= stjude-mygpt
APP_DIR := app

# Phase-2 eval corpora: one docs-volume folder per corpus (= the agent's custom_inputs.corpus).
DOCS_VOLUME := dbfs:/Volumes/structural_biology/mygpt/docs
EVAL_INPUTS := reference/MyGPT-evaluations
# St. Jude's private ground truths (gitignored) -> the restricted eval_private UC volume.
EVAL_PRIVATE := reference/MyGPT-evaluations-private-main/ground truths
PRIVATE_VOLUME := dbfs:/Volumes/structural_biology/mygpt/eval_private
KINASE_PDFS ?=   # dir of St. Jude's kinase PDFs, named <pubmed_id>.pdf (not in the submodule)

.DEFAULT_GOAL := help

.PHONY: help init submodules submodules-update install dev build typecheck lint test validate deploy clean eval-corpus eval-private eval-test

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

init: submodules install ## First-time setup: fetch submodules + install app deps

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

validate: ## Validate the app before deploy (PROFILE=<name>)
	cd $(APP_DIR) && databricks apps validate --profile $(PROFILE)

deploy: ## Deploy the app to the workspace (PROFILE=<name>)
	cd $(APP_DIR) && databricks apps deploy --profile $(PROFILE)

clean: ## Remove app build artifacts + node_modules
	cd $(APP_DIR) && pnpm run clean

eval-corpus: ## Upload the Phase-2 eval corpora to the docs volume (PROFILE=<name>, KINASE_PDFS=<dir>)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/health_policies/inputs/French_10_pdfs" $(DOCS_VOLUME)/health_policies_fr --profile $(PROFILE)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/health_policies/inputs/Spanish_10_pdfs" $(DOCS_VOLUME)/health_policies_es --profile $(PROFILE)
	databricks fs cp -r --overwrite "$(EVAL_INPUTS)/BioASQ/inputs/pdfs" $(DOCS_VOLUME)/biomed --profile $(PROFILE)
	@if [ -n "$(KINASE_PDFS)" ]; then \
	  databricks fs cp -r --overwrite "$(KINASE_PDFS)" $(DOCS_VOLUME)/kinase_literature --profile $(PROFILE); \
	else echo "KINASE_PDFS not set — skipping kinase_literature (St. Jude to provide the PDFs)"; fi

eval-private: ## Upload St. Jude's private ground truths to the restricted eval_private volume (PROFILE=<name>)
	@test -d "$(EVAL_PRIVATE)" || (echo "missing $(EVAL_PRIVATE) (St. Jude's private ground truths)"; exit 1)
	databricks fs cp -r --overwrite "$(EVAL_PRIVATE)" $(PRIVATE_VOLUME) --profile $(PROFILE)

eval-test: ## Unit-test the eval harness, agent scoring / provided-sources, Supervisor + tools (no workspace needed)
	uv run --no-project --with-requirements eval/requirements.txt pytest -q eval/ agents/rag_agent/test_scoring.py agents/rag_agent/test_provided_sources.py agents/supervisor/test_supervisor.py agents/tools/test_tools.py
