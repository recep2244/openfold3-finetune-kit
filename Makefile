.DEFAULT_GOAL := help
SHELL := bash

.PHONY: help verify lint fmt test docs docs-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

verify: ## Validate the install and run a smoke prediction
	bash scripts/verify_setup.sh

lint: ## shellcheck + yamllint + ruff + notebook validation
	@echo ">> shellcheck"; shellcheck scripts/*.sh docker/entrypoint.sh
	@echo ">> bash -n";    for s in scripts/*.sh docker/entrypoint.sh; do bash -n "$$s"; done
	@echo ">> yamllint";   yamllint -d "{extends: relaxed, rules: {line-length: disable}}" configs/ .github/
	@echo ">> ruff";       ruff check .
	@echo ">> notebooks";  python -c "import glob,nbformat; [nbformat.read(f,4) for f in glob.glob('notebooks/*.ipynb')]"

fmt: ## Format Python with ruff
	ruff format .
	ruff check --fix .

test: ## Run shell script tests (Bats)
	bats tests/

docs: ## Serve the documentation site locally
	mkdocs serve

docs-build: ## Build the documentation site (strict)
	mkdocs build --strict

clean: ## Remove local build/output artifacts
	rm -rf site .ruff_cache **/__pycache__ .ipynb_checkpoints _setup*.log _smoke*.log
