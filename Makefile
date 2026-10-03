.PHONY: install test lint format schema audit demo

install:
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check src tests --fix

schema:
	uv run python -m profile_builder schema --out schema/company_brain.schema.json

audit:
	uv run pip-audit --strict || true

demo:
	uv run python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform"
