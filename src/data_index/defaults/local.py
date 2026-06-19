import pathlib

import prefect

import data_index
from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config import (
    IcebergTableConfig,
    IcebergTableScanConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import LiveS3InventorySource, ParquetInventorySource
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
    NetCDFExtractor,
    UnstructuedNetCDFExtractor,
)
from data_index.structured_metadata import StructuredMetadata
from data_index.structured_sink import StructuredParquetSink, StructuredS3TableSink
from data_index.unstructured_metadata import (
    DiskCachedUnstructuredMetadata,
    InMemoryUnstructuredMetadata,
)
from data_index.unstructured_sink import (
    UnstructuredParquetSink,
    UnstructuredS3TableSink,
)

# --- General config ---
REGION = "ap-southeast-2"
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB

# --- Local config ---
BATCH_SIZE = 1_000
MAX_WORKERS = 2  # concurrent batches (limits RAM/CPU pressure)
S5CMD_WORKERS = 8  # s5cmd defaults to 256 — cap it for local runs
TRANSFORM_WORKERS = (
    12  # transform threads per batch (total = MAX_WORKERS × TRANSFORM_WORKERS)
)

# --- Live Inventory Source config
s3_metadata_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
)

inventory_table_config = IcebergTableConfig(
    catalog_config=s3_metadata_catalog_config,
    namespace="b_imos-data",
    table_name="inventory",
)

inventory_table_scan_config = IcebergTableScanConfig(
    row_filter="key LIKE 'IMOS/%'",
)

live_inventory_source = LiveS3InventorySource(
    table_config=inventory_table_config,
    table_scan_config=inventory_table_scan_config,
    path=pathlib.Path(".extract/s3_metadata"),
    skip_if_exists=True,
)

# --- Partitioner config ---
partitioner = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=50 * 1024**3,
)

# --- File fetcher ---
fetcher = S5CMDFetcher(num_workers=S5CMD_WORKERS, anon=True)

# --- Metadata extractor ---
extractor = AttributeNetCDFExtractor()

# --- Sink config ---
data_index_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)

structured_metadata_table_config = IcebergTableConfig(
    catalog_config=data_index_catalog_config,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

structured_sink = StructuredS3TableSink(
    iceberg_table_config=structured_metadata_table_config,
)

unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=data_index_catalog_config,
    namespace="data_index",
    table_name="unstructured_metadata",
)

unstructured_sink = UnstructuredS3TableSink(
    iceberg_table_config=unstructured_metadata_table_config,
)


@prefect.flow
def run_index_local(
    inventory_source: LiveS3InventorySource
    | ParquetInventorySource = live_inventory_source,
    partitioner: GreedyBatchPartitioner = partitioner,
    fetcher: S3Fetcher | S5CMDFetcher | ThresholdFileFetcher = fetcher,
    extractor: NetCDFExtractor
    | UnstructuedNetCDFExtractor
    | AttributeNetCDFExtractor = extractor,
    structured_sink: StructuredParquetSink | StructuredS3TableSink = structured_sink,
    unstructured_sink: UnstructuredParquetSink
    | UnstructuredS3TableSink = unstructured_sink,
    metadata_factory: InMemoryUnstructuredMetadata
    | DiskCachedUnstructuredMetadata
    | None = None,
    transform_max_workers: int | None = TRANSFORM_WORKERS,
):

    data_index.orchestrate.with_options(
        task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=MAX_WORKERS)
    )(
        inventory_source,
        partitioner,
        fetcher,
        extractor,
        structured_sink,
        unstructured_sink,
        metadata_factory,
        transform_max_workers,
    )


if __name__ == "__main__":
    run_index_local()
