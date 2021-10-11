rm -rf logs/
isort .
black --line-length 120 .
rm code.zip
find . \( -name __pycache__ -o -name "*.pyc" \) -delete
find . \
  -path '*/data' -prune -o \
  -path '*/wandb' -prune -o \
  -path '*/*.zip' -prune -o \
  -path '*/.*' -type d -prune -o \
  -type f -print | zip code.zip -@