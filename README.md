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

### Opt in to DynamoDB for structured and unstructured metadata

`DynamoDBSink` is available as an alternative metadata sink. It is keyed by `hash` and
is latest-write-wins for duplicate object versions. Use separate tables per metadata type.

```python
from data_index.runners.index import index
from data_index.runners.defaults import (
    DEAD_LETTER_TABLE_SINK,
    INVENTORY_SOURCE,
    BATCH_PARTITIONER,
    FILE_FETCHER,
    METADATA_EXTRACTOR,
)
from data_index.sink import DynamoDBSink

index(
    inventory_source=INVENTORY_SOURCE,
    partitioner=BATCH_PARTITIONER,
    fetcher=FILE_FETCHER,
    extractor=METADATA_EXTRACTOR,
    structured_sink=DynamoDBSink(
        table_name="data-index-structured-metadata-v6",
        region_name="ap-southeast-2",
    ),
    unstructured_sink=DynamoDBSink(
        table_name="data-index-unstructured-metadata-v4",
        region_name="ap-southeast-2",
    ),
    dead_letter_sink=DEAD_LETTER_TABLE_SINK,
)
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
Set up the development environement with the `make init` command:
```bash
make init
```
