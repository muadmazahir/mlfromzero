test:
	poetry run pytest tests/ -vv

lint-fix:
	poetry run ruff check --fix tests/ mlfromzero/

lint:
	poetry run ruff check tests/ mlfromzero/

format:
	poetry run ruff format tests/ mlfromzero/

format-check:
	poetry run ruff format --check tests/ mlfromzero/

all: lint format sort-imports test
