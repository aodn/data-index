# --------------------------------------------------------------------
# --- Base Target ---
# --------------------------------------------------------------------
FROM prefecthq/prefect-aws:0.7.7-python3.12-prefect3.6.28 AS base

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/

# System libraries required by netcdf4, h5py, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# --- App Target ---
# --------------------------------------------------------------------
FROM base AS app
WORKDIR /app

# Copy everything needed for the application install at once
COPY requirements.txt constraints.txt dist/*.whl ./

# Install app
RUN uv pip install --system -c constraints.txt -r requirements.txt *.whl

# --------------------------------------------------------------------
# --- Test Target ---
# --------------------------------------------------------------------
FROM app AS test

# Copy over the test folder and structural files needed to export
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# 1. Export the dev group packages from the lockfile into a temporary text file
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN uv export --group dev --no-emit-project --format requirements-txt --output-file dev-reqs.txt

# 2. Install them into the system space alongside your existing application
RUN uv pip install --system -r dev-reqs.txt

# 3. Clean up the file and execute tests
RUN pytest tests/ -v