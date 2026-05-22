FROM python:3.12-slim

# Install uv (matches the local toolchain)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System libraries required by netcdf4, h5py, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first — layer cache stays valid until these change
COPY pyproject.toml uv.lock README.md ./

# Install all dependencies from the lock file (no project install yet)
RUN uv sync --frozen --no-install-project

# Copy the package source
COPY src/ ./src/

# Install the data-index package itself (still using frozen lock)
RUN uv sync --frozen

# Put the venv on PATH for the dask worker entry-point
ENV PATH="/app/.venv/bin:$PATH"

# Dask workers are launched by FargateCluster; no CMD needed
