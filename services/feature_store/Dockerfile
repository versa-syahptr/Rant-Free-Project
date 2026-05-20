FROM python:3.12-slim

WORKDIR /app/feature_store

# Install deps — cached as long as pyproject.toml doesn't change
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Dataset first (~455 MB, rarely changes) — keeps this layer cached across code edits
COPY data/dataset.parquet ./data/

# Source code — changes more often, goes after the large dataset layer
COPY *.py *.yaml ./

CMD ["python", "bootstrap.py"]
