# MASS Makefile
# Windows users: use `python run.py` instead

.PHONY: install test test-cov run run-mock cli diagnose lint clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=agent --cov=api --cov-report=html --cov-report=term

run:
	python app.py

run-mock:
	python run.py --mock

cli:
	python mass_cli/commands.py --mock

diagnose:
	python mass_cli/commands.py --mock diagnose $(code)

batch:
	python mass_cli/commands.py --mock batch $(codes)

lint:
	python -m flake8 agent/ api/ tests/ --max-line-length=120 --extend-ignore=E501,W503

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
