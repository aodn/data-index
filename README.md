# Data Index

A pipeline that ingests CF-compliant NetCDF files from S3, extracts metadata, and stores it for discovery and later analysis.

## Pipeline

```
[ INJECTED DEPENDENCIES ]
  ├── InventorySource
  ├── BatchPartitioner
  ├── FileFetcher
  ├── MetadataExtractor
  ├── StructuredSink 
  └── UnstructuredSink
         │
         ▼
 ┌────────────────────────────────────────────────────────┐
 │ Orchestrator (Prefect Flow)                            │
 ├────────────────────────────────────────────────────────┤
 │                                                        │
 │  1. [ Sinks.provision() ]                              │
 │                                                        │
 │  2. [ InventorySource.inventory() ] ──► Full Corpus    │
 │                                           │            │
 │  3. [ BatchPartitioner.partition() ] ◄────┘            │
 │            │                                           │
 │            ▼                                           │
 │     [ Split Batches ]                                  │
 │            │                                           │
 └────────────┼───────────────────────────────────────────┘
              │
              │ Dispatch concurrent workers
              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Concurrently Executed Batch Process                      │
 ├──────────────────────────────────────────────────────────┤
 │                                                          │
 │    extract() ◄───────── [ FileFetcher.fetch() ]          │
 │        │                                                 │
 │        ▼                                                 │
 │   transform() ◄──────── [ MetadataExtractor.extract() ]  │
 │        │                                                 │
 │        ▼                                                 │
 │  ExtractionResult(structured + unstructured + status)    │
 │                                                          │
 │        │                                                 │
 │        ▼                                                 │
 │     load()                                               │
 │        ├──► [ StructuredSink.sink() ]   ──► store        │
 │        └──► [ UnstructuredSink.sink() ] ──► store        │
 │                                                          │
 └──────────────────────────────────────────────────────────┘
```

## Running locally

Start a local Prefect server:
```bash
uv run prefect server start
```

Run a local test against a sampled inventory:
```bash
uv run cluster-local
```

Run against AWS Fargate (builds + pushes Docker image to ECR):
```bash
uv run cluster-fargate
```

## Reading results

```python
from pyiceberg.catalog.sql import SqlCatalog
import polars

catalog = SqlCatalog(
    "data-index",
    uri="sqlite:///.load/orchestrate-test/catalog.db",
    warehouse=".load/orchestrate-test",
)

df = polars.from_arrow(
    catalog.load_table(("structured-metadata", "test")).scan().to_arrow()
)
```

## Development

```bash
uv sync --group dev
uv run pre-commit install   # install ruff check + format hooks
uv run pytest tests/ -v
```
