rm -rf logs/
isort .
black --line-length 120 .
vulture . --min-confidence 100
flake8 .
