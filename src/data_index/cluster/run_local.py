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

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner
from data_index.cluster.orchestrate import orchestrate
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.inventory_source.parquet import ParquetInventorySource
from data_index.metadata_extractor import UnstructuedNetCDFExtractor
from data_index.structured_sink import StructuredParquetSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.unstructured_sink import UnstructuredParquetSink

# --- Config ---
LIMIT = 100_000          # total files to process
BATCH_SIZE = 10_000      # files per batch → 4 batches
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

    orchestrate(
        inventory_source=ParquetInventorySource(path=INVENTORY_PATH),
        partitioner=GreedyBatchPartitioner(
            max_files=BATCH_SIZE,
            max_bytes=50 * 1024 ** 3,  # 50 GB
        ),
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=THRESHOLD_BYTES,
            disk_fetcher=S5CMDFetcher(),
            cloud_fetcher=S3Fetcher(block_size=5 * 1024 ** 2),
        ),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredParquetSink(
            path=OUT_DIR / "structured_metadata.parquet",
        ),
        unstructured_sink=UnstructuredParquetSink(
            path=OUT_DIR / "unstructured_metadata.parquet",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
    )


if __name__ == "__main__":
    main()
