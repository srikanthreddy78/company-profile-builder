.PHONY: install test lint format schema audit docker demo

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

# Advisory: findings are printed but never fail the target (CI does the same).
audit:
	uv run pip-audit --strict || true

docker:
	docker build -t profile-builder .

demo:
	uv run python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform"
