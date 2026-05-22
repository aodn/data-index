"""
Test script: run the Orchestrator locally against a live S3 inventory.

Usage:
    uv run cluster-local

Fetches the live inventory from the S3 Tables Iceberg catalog, partitions
into batches of BATCH_SIZE, and runs extract → transform → load via the
orchestrate flow using a local ThreadPoolTaskRunner.

Output lands in .load/orchestrate-test/.
"""

import pathlib

import prefect.task_runners

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner
from data_index.catalog_config import S3TablesCatalogConfig, SqliteCatalogConfig
from data_index.cluster.orchestrate import orchestrate
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_table_config import IcebergTableConfig
from data_index.inventory_source.live_s3 import LiveS3InventorySource
from data_index.metadata_extractor import UnstructuedNetCDFExtractor
from data_index.s3_metadata.extract import TableScanConfig
from data_index.structured_sink import StructuredS3TableSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.unstructured_sink import UnstructuredS3TableSink

# --- Config ---
BATCH_SIZE = 1_000  # files per batch
MAX_WORKERS = 8  # concurrent batches (limits RAM/CPU pressure)
S5CMD_WORKERS = 8  # s5cmd defaults to 256 — cap it for local runs
TRANSFORM_WORKERS = (
    4  # transform threads per batch (total = MAX_WORKERS × TRANSFORM_WORKERS)
)
OUT_DIR = pathlib.Path(".load/orchestrate-test")
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Catalog for output sinks (local SQLite) ---
_sink_catalog_config = SqliteCatalogConfig(
    uri=f"sqlite:///{OUT_DIR}/catalog.db",
    warehouse=str(OUT_DIR.resolve()),
)

# --- Inventory source (live S3 Tables) ---
_inventory_table_config = IcebergTableConfig(
    catalog_config=S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
    ),
    namespace="b_imos-data",
    table_name="inventory",
)


def main() -> None:
    sink_catalog = _sink_catalog_config.build()

    orchestrate.with_options(
        task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=MAX_WORKERS)
    )(
        inventory_source=LiveS3InventorySource(
            table_config=_inventory_table_config,
            table_scan_config=TableScanConfig(row_filter="key LIKE 'IMOS/%'"),
            path=pathlib.Path(".extract/s3_metadata"),
            skip_if_exists=True,
        ),
        partitioner=GreedyBatchPartitioner(
            max_files=BATCH_SIZE,
            max_bytes=50 * 1024**3,  # 50 GB
        ),
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=THRESHOLD_BYTES,
            disk_fetcher=S5CMDFetcher(num_workers=S5CMD_WORKERS, anon=True),
            cloud_fetcher=S3Fetcher(block_size=5 * 1024**2),
        ),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredS3TableSink(
            catalog=sink_catalog,
            namespace="structured-metadata",
            table_name="test",
        ),
        unstructured_sink=UnstructuredS3TableSink(
            catalog=sink_catalog,
            namespace="unstructured-metadata",
            table_name="test",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
        transform_max_workers=TRANSFORM_WORKERS,
    )


if __name__ == "__main__":
    main()
