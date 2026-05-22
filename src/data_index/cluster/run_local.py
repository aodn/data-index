"""
Test script: run the Orchestrator locally against a small inventory sample.

Usage:
    uv run python src/data_index/cluster/run_local.py

Reads from .extract/s3_metadata, samples LIMIT small NetCDF files, partitions
into batches of BATCH_SIZE, and runs extract → transform → load via the
orchestrate flow using a local ThreadPoolTaskRunner.

Output lands in .load/orchestrate-test/.
"""

import pathlib

import polars
import prefect.task_runners
from pyiceberg.catalog.sql import SqlCatalog

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner

from data_index.cluster.orchestrate import orchestrate
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.inventory_source.parquet import ParquetInventorySource
from data_index.metadata_extractor import UnstructuedNetCDFExtractor
from data_index.structured_sink import StructuredParquetSink, StructuredS3TableSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.unstructured_sink import UnstructuredParquetSink, UnstructuredS3TableSink

# --- Config ---
LIMIT = 16_000             # total files to process
BATCH_SIZE = 1_000         # files per batch
MAX_WORKERS = 8         # concurrent batches (limits RAM/CPU pressure)
S5CMD_WORKERS = 8       # s5cmd defaults to 256 — cap it for local runs
TRANSFORM_WORKERS = 4   # transform threads per batch (total = MAX_WORKERS × TRANSFORM_WORKERS)
OUT_DIR = pathlib.Path(".load/orchestrate-test")
INVENTORY_PATH = OUT_DIR / "inventory.parquet"
THRESHOLD_BYTES = 10 * 1024 ** 2  # 10 MB

OUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_inventory() -> None:
    """Sample LIMIT small .nc files from the local s3_metadata cache and write
    a prepared inventory (s3_uri, size) parquet for ParquetInventorySource."""
    df = (
        polars.scan_parquet(".extract/s3_metadata")
        .filter(
            polars.col("key").str.ends_with(".nc"),
            polars.col("size").le(1024 ** 2),  # ≤ 1 MB for a fast local test
        )
        .select(
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/",
            ).alias("s3_uri"),
            polars.col("size"),
        )
        .collect()
        .sample(LIMIT, seed=42)
    )
    df.write_parquet(INVENTORY_PATH)
    print(f"Prepared inventory: {len(df)} files → {INVENTORY_PATH}")


def main() -> None:
    prepare_inventory()

    catalog = SqlCatalog(
        "data-index",
        uri=f"sqlite:///{OUT_DIR}/catalog.db",
        warehouse=str(OUT_DIR.resolve()),
    )

    orchestrate.with_options(
        task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=MAX_WORKERS)
    )(
        inventory_source=ParquetInventorySource(path=INVENTORY_PATH),
        partitioner=GreedyBatchPartitioner(
            max_files=BATCH_SIZE,
            max_bytes=50 * 1024 ** 3,  # 50 GB
        ),
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=THRESHOLD_BYTES,
            disk_fetcher=S5CMDFetcher(num_workers=S5CMD_WORKERS, anon=True),
            cloud_fetcher=S3Fetcher(block_size=5 * 1024 ** 2),
        ),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredS3TableSink(
            catalog=catalog,
            namespace="structured-metadata",
            table_name="test",
        ),
        unstructured_sink=UnstructuredS3TableSink(
            catalog=catalog,
            namespace="unstructured-metadata",
            table_name="test",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
        transform_max_workers=TRANSFORM_WORKERS,
    )


if __name__ == "__main__":
    main()
