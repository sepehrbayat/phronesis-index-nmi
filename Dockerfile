# ===========================================================
#  Phronesis Index — Reproducibility Container
# ===========================================================
# Build:  docker build -t phronesis .
# Run:    docker run --rm phronesis pytest
# Full:   docker run --rm phronesis bash scripts/reproduce_all.sh --smoke
# ===========================================================
FROM python:3.9-slim

LABEL maintainer="Sepehr Bayat <sepehrbayat@hooshex.com>"
LABEL description="Phronesis Index — Spectral Sheaf Heuristics for Consistency Detection"

WORKDIR /app

# System deps (for scipy / matplotlib backends)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY env/requirements.txt env/requirements.txt
RUN pip install --no-cache-dir -r env/requirements.txt

# Copy repo
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e ".[experiments,dev]"

# Default: run tests
CMD ["pytest", "-v", "--tb=short"]
