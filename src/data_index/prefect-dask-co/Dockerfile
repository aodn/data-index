FROM prefecthq/prefect:3.6.27-python3.12

COPY pyproject.toml uv.lock ./

RUN uv pip install --system -r pyproject.toml