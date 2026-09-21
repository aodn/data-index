import pathlib
import typing

import prefect

import data_index.analysis.tables as analysis_tables
import data_index.analysis.warehouse as analysis_warehouse
import data_index.runners.defaults as runners_defaults
from data_index.file_fetcher import LocalFetcher
from data_index.inventory_source import LocalGlobInventorySource
from data_index.runners.index import index as index_flow
from data_index.runners.task_runner import (
    ProcessPoolRunnerConfig,
    ThreadPoolRunnerConfig,
)
from data_index.runners.types import (
    BatchPartitioner,
    FileFetcher,
    InventorySource,
    MetadataExtractor,
    MetadataSink,
)
from data_index.sink import IcebergTableSink

DATASET: typing.Literal["station_lucinda_jetty_daily_wetlabs_bb9"] = (
    "station_lucinda_jetty_daily_wetlabs_bb9"
)
LOCAL_DATASET_ROOT_PATH = pathlib.Path("/Volumes/4tb-0/imos-data/IMOS/SRS/OC/LJCO")
LOCAL_DATASET_GLOB_PATTERN = "**/*.nc"
LOCAL_BUCKET = "imos-data"
LOCAL_VERSION_ID = "__LOCAL__"

# --- Dataset-scoped local inventory + fetch config ---
INVENTORY_SOURCE = LocalGlobInventorySource(
    root_path=LOCAL_DATASET_ROOT_PATH,
    glob_pattern=LOCAL_DATASET_GLOB_PATTERN,
    bucket=LOCAL_BUCKET,
    local_version_id=LOCAL_VERSION_ID,
)
FILE_FETCHER = LocalFetcher(local_version_id=LOCAL_VERSION_ID)

# --- Shared pipeline defaults ---
BATCH_PARTITIONER = runners_defaults.BATCH_PARTITIONER
METADATA_EXTRACTOR = runners_defaults.METADATA_EXTRACTOR
TASK_RUNNER_CONFIG = runners_defaults.TASK_RUNNER_CONFIG
BATCH_MAX_WORKERS = runners_defaults.BATCH_MAX_WORKERS

LOCAL_WAREHOUSE = analysis_warehouse.ANALYSIS_LOCAL_WAREHOUSE

STRUCTURED_TABLE_CONFIG = analysis_tables.LOCAL_STRUCTURED_METADATA_TABLE
STRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind="structured",
    iceberg_table_config=STRUCTURED_TABLE_CONFIG,
    partition_column=runners_defaults.STRUCTURED_TABLE_SINK.partition_column,
)

UNSTRUCTURED_TABLE_CONFIG = analysis_tables.LOCAL_UNSTRUCTURED_METADATA_TABLE
UNSTRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind="unstructured",
    iceberg_table_config=UNSTRUCTURED_TABLE_CONFIG,
    partition_column=runners_defaults.UNSTRUCTURED_TABLE_SINK.partition_column,
)

DEAD_LETTER_TABLE_CONFIG = analysis_tables.LOCAL_DEAD_LETTER_TABLE
DEAD_LETTER_TABLE_SINK = IcebergTableSink(
    schema_kind="dead_letter",
    iceberg_table_config=DEAD_LETTER_TABLE_CONFIG,
    partition_column=runners_defaults.DEAD_LETTER_TABLE_SINK.partition_column,
)


@prefect.flow
def index(
    inventory_source: InventorySource = INVENTORY_SOURCE,
    partitioner: BatchPartitioner = BATCH_PARTITIONER,
    fetcher: FileFetcher = FILE_FETCHER,
    extractor: MetadataExtractor = METADATA_EXTRACTOR,
    structured_sink: MetadataSink = STRUCTURED_TABLE_SINK,
    unstructured_sink: MetadataSink = UNSTRUCTURED_TABLE_SINK,
    dead_letter_sink: MetadataSink = DEAD_LETTER_TABLE_SINK,
    index_batch_flow_name: str = "index-batch",
    index_batch_deployment_name: str = "index-batch",
    task_runner_config: ProcessPoolRunnerConfig
    | ThreadPoolRunnerConfig = TASK_RUNNER_CONFIG,
    batch_max_workers: int | None = max_workers
    if (max_workers := BATCH_MAX_WORKERS - 4) > 0
    else None,
):
    LOCAL_WAREHOUSE.mkdir(parents=True, exist_ok=True)
    return index_flow(
        inventory_source=inventory_source,
        partitioner=partitioner,
        fetcher=fetcher,
        extractor=extractor,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
        dead_letter_sink=dead_letter_sink,
        index_batch_flow_name=index_batch_flow_name,
        index_batch_deployment_name=index_batch_deployment_name,
        task_runner_config=task_runner_config,
        batch_max_workers=batch_max_workers,
    )
