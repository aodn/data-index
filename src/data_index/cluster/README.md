# Prefect Dask CO

## Install

### Authenticate Docker
```bash
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 704910415367.dkr.ecr.ap-southeast-2.amazonaws.com
```

## Run
```bash
uv run main.py
```