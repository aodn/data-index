FROM prefecthq/prefect-aws:0.7.7-python3.12-prefect3.6.28 AS base

# Building version requirements
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
ENV HATCH_BUILD_VERSION=${VERSION}
ENV UV_SYSTEM_PYTHON=1

# System libraries required by netcdf4, h5py, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first — layer cache stays valid until these change
COPY pyproject.toml uv.lock README.md ./

# Explicitly install your build backend globally first
RUN uv pip install hatchling

# Install all dependencies from the lock file (no project install yet)
RUN uv sync --frozen --no-install-project

# Copy the package source
COPY src/ ./src/

# Install the data-index package itself (still using frozen lock)
RUN uv sync --frozen --no-build-isolation

# Install and run tests
FROM base AS test
COPY tests/ ./tests/
RUN uv sync --frozen --group dev --no-build-isolation
RUN uv run pytest tests/ -v

# Expose only base
FROM base AS production
