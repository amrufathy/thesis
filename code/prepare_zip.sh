rm -rf logs/
isort .
black --line-length 120 .
rm code.zip
find . \( -name __pycache__ -o -name "*.pyc" \) -delete
#find . -path '*/.*' -prune -o -type f -print | zip code.zip -@
find . -path '*/.*' -prune -o -path '*.csv' -prune -o -type f -print | zip code.zip -@
