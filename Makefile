# MyGPT on Databricks — POC monorepo
# Common tasks. Run `make help` to list targets.
# Override the target workspace with PROFILE=<name> (default: stjude-mygpt).
# The app standardizes on pnpm — see app/package.json `packageManager`
# (pnpm@11.0.8, provisioned via corepack) + app/pnpm-workspace.yaml. Don't run
# `npm install` in app/ (it would re-create package-lock.json and diverge from
# the pnpm-lock.yaml the Databricks Apps build compute installs from).

PROFILE ?= stjude-mygpt
APP_DIR := app

.DEFAULT_GOAL := help

.PHONY: help init submodules submodules-update install dev build typecheck lint test validate deploy clean

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
