.PHONY: help install install-dev format lint test clean setup-precommit

help: ## Hiển thị help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Cài đặt production dependencies
	pip install -r requirements.txt

install-dev: ## Cài đặt development dependencies
	pip install -r requirements-dev.txt
	npm install

format: ## Format code Python và JavaScript
	@echo "Formatting Python code..."
	isort . --profile black --line-length 100
	black . --line-length 100
	@echo "Formatting JavaScript code..."
	npm run format:js

format-check: ## Kiểm tra format (không sửa)
	@echo "Checking Python code format..."
	isort . --profile black --line-length 100 --check-only
	black . --line-length 100 --check
	@echo "Checking JavaScript code format..."
	npm run format:js:check

lint: ## Lint code Python và JavaScript
	@echo "Linting Python code..."
	flake8 . --max-line-length=100 --extend-ignore=E203,W503,E226,E722,F401,F541,C901 --max-complexity=30 --statistics
	@echo "Linting JavaScript code..."
	@if [ -f package.json ]; then npm run lint:js; else echo "No JS files to lint"; fi

lint-fix: ## Lint và tự động fix JavaScript
	npm run lint:js:fix

check: format-check lint ## Chạy tất cả checks (format + lint)

test: ## Chạy tests
	pytest

clean: ## Xóa cache và build files
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info

setup-precommit: ## Setup pre-commit hooks
	pre-commit install
	pre-commit run --all-files

all: install-dev format lint ## Cài đặt, format và lint tất cả
