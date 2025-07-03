test:
	poetry run pytest tests -vv

lint-fix:
	poetry run ruff check --fix tests/ mlfromzero/

lint:
	poetry run ruff check tests/ mlfromzero/