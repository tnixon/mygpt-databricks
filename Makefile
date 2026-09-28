# MyGPT on Databricks — POC monorepo
# Common tasks. Run `make help` to list targets.
# Override the target workspace with PROFILE=<name> (default: stjude-mygpt).

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

install: ## Install app dependencies
	cd $(APP_DIR) && npm install

dev: ## Run the app locally (http://localhost:8000)
	cd $(APP_DIR) && npm run dev

build: ## Build the app
	cd $(APP_DIR) && npm run build

typecheck: ## Type-check the app
	cd $(APP_DIR) && npm run typecheck

lint: ## Lint the app
	cd $(APP_DIR) && npm run lint

test: ## Run app tests
	cd $(APP_DIR) && npm test

validate: ## Validate the app before deploy (PROFILE=<name>)
	cd $(APP_DIR) && databricks apps validate --profile $(PROFILE)

deploy: ## Deploy the app to the workspace (PROFILE=<name>)
	cd $(APP_DIR) && databricks apps deploy --profile $(PROFILE)

clean: ## Remove app build artifacts + node_modules
	cd $(APP_DIR) && npm run clean
