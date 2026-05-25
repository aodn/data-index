FROM prefecthq/prefect-aws:0.7.7-python3.12-prefect3.6.28

# Install uv for fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/

# System libraries required by netcdf4, h5py, scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
        libhdf5-dev \
        libnetcdf-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# -- Build app --
WORKDIR /app

# Install dependencies from the pinned lock file, capture constraints in constraints.txt
COPY requirements.txt ./
COPY constraints.txt ./
RUN uv pip install --system --build-constraints constraints.txt --requirements requirements.txt
    
# Copy the package source and install data-index itself
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
COPY src/ ./src/
COPY pyproject.toml README.md ./
RUN uv pip install --system --no-deps .
