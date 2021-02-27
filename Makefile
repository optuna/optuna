TOX := $(shell command -v tox)

.DEFAULT_GOAL := help

.PHONY: lint
lint: ## Run linter
ifeq ($(TOX),)
	black . --check --diff
	flake8 .
	isort . --check --diff
	mypy .
	blackdoc . --check --diff
else 
	$(TOX) -e black
	$(TOX) -e flake8
	$(TOX) -e isort
	$(TOX) -e mypy
	$(TOX) -e blackdoc
endif

.PHONY: fmt
fmt: ## Run formatter
ifeq ($(TOX),)
	black .
	isort .
	blackdoc .
else 
	$(TOX) -e black-fmt
	$(TOX) -e isort-fmt
	$(TOX) -e blackdoc-fmt
endif

.PHONY: help
help: ## Show help text
	@echo "Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

