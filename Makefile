.PHONY: setup sync lint fmt type test test-fast cov check clean

setup:        ## İlk kurulum
	bash setup.sh

sync:         ## Bağımlılıkları güncelle
	uv sync --all-extras

lint:
	uv run ruff check .

fmt:
	uv run ruff check . --fix
	uv run ruff format .

type:
	uv run mypy src

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow" -x -q

cov:
	uv run pytest --cov --cov-report=term-missing

check: lint type test   ## Commit öncesi zorunlu (CLAUDE.md §2)
	uv run ruff format --check .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
