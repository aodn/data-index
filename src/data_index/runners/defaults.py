import pathlib

import data_index.protocols
from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import (
    ObstoreFetcher,
)
from data_index.iceberg_config import (
    IcebergTableConfig,
    IcebergTableScanConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import (
    IcebergTableInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.runners.task_runner import (
    ThreadPoolRunnerConfig,
)
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata
from data_index.sink import (
    IcebergTableSink,
)

# --- General config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
BATCH_SIZE = 1_000
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2 * 10  # 100 MB
S5CMD_WORKERS = 8


# --- Live Inventory Source config
_S3_METADATA_CATALOG_CONFIG = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
)

_INVENTORY_TABLE_CONFIG = IcebergTableConfig(
    catalog_config=_S3_METADATA_CATALOG_CONFIG,
    namespace="inventory",
    table_name="live",
)

INVENTORY_SOURCE = IcebergTableInventorySource(
    table_config=_INVENTORY_TABLE_CONFIG,
    table_scan_config=IcebergTableScanConfig(
        row_filter="key LIKE 'IMOS/SOOP/%' OR key LIKE 'IMOS/AATAMS/%' OR key LIKE 'IMOS/ANMN/%' OR key LIKE 'IMOS/FAIMMS/%' OR key LIKE 'IMOS/OceanCurrent/%' OR key LIKE 'IMOS/DWM/%' OR key LIKE 'IMOS/AUV/%' OR key LIKE 'IMOS/COASTAL-WAVE-BUOYS/%' OR key LIKE 'IMOS/NTP/%' OR key LIKE 'IMOS/ANFOG/%' OR key LIKE 'IMOS/eMII/%'",
        selected_fields=["bucket", "key", "version_id", "size"],
    ),
)

# --- Partitioner config ---
BATCH_PARTITIONER = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=10 * 1024**3,
)

# --- File fetcher ---
FILE_FETCHER = ObstoreFetcher()

# --- Metadata extractor ---
METADATA_EXTRACTOR = AttributeNetCDFExtractor()

# --- Sink config ---
_DATA_INDEX_CATALOG_CONFIG = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)
_STRUCTURED_METADATA_TABLE_CONFIG = IcebergTableConfig(
    catalog_config=_DATA_INDEX_CATALOG_CONFIG,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

STRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind="structured",
    iceberg_table_config=_STRUCTURED_METADATA_TABLE_CONFIG,
    partition_column="facility",
)

_UNSTRUCTURED_METADATA_TABLE_CONFIG = IcebergTableConfig(
    catalog_config=_DATA_INDEX_CATALOG_CONFIG,
    namespace="data_index",
    table_name=f"unstructured_metadata_v{UnstructuredMetadata.SCHEMA_VERSION}",
)

UNSTRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind="unstructured",
    iceberg_table_config=_UNSTRUCTURED_METADATA_TABLE_CONFIG,
    partition_column="facility",
)

_DEAD_LETTER_TABLE_CONFIG = IcebergTableConfig(
    catalog_config=_DATA_INDEX_CATALOG_CONFIG,
    namespace="data_index",
    table_name=f"dead_letter_v{data_index.protocols.DeadLetter.SCHEMA_VERSION}",
)

DEAD_LETTER_TABLE_SINK = IcebergTableSink(
    schema_kind="dead_letter",
    iceberg_table_config=_DEAD_LETTER_TABLE_CONFIG,
)

# --- Runtime Config ---
TASK_RUNNER_CONFIG = ThreadPoolRunnerConfig()
