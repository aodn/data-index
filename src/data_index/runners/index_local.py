import pathlib

import prefect

import data_index.runners.defaults as runners_defaults
from data_index.file_fetcher import LocalFetcher
from data_index.iceberg_config import IcebergTableConfig, SqliteCatalogConfig
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

# --- Local inventory + fetch config ---
LOCAL_ROOT_PATH = pathlib.Path("/Volumes/4tb-0/dev/data/Argo/")
LOCAL_GLOB_PATTERN = "**/*_prof.nc"
LOCAL_BUCKET = "__LOCAL_VOLUME__"
LOCAL_VERSION_ID = "__LOCAL__"

INVENTORY_SOURCE = LocalGlobInventorySource(
    root_path=LOCAL_ROOT_PATH,
    glob_pattern=LOCAL_GLOB_PATTERN,
    bucket=LOCAL_BUCKET,
    local_version_id=LOCAL_VERSION_ID,
    max_files=10,
)

FILE_FETCHER = LocalFetcher(
    bucket=LOCAL_BUCKET,
    local_version_id=LOCAL_VERSION_ID,
)

# --- Shared pipeline defaults ---
BATCH_PARTITIONER = runners_defaults.BATCH_PARTITIONER
METADATA_EXTRACTOR = runners_defaults.METADATA_EXTRACTOR
TASK_RUNNER_CONFIG = runners_defaults.TASK_RUNNER_CONFIG
BATCH_MAX_WORKERS = runners_defaults.BATCH_MAX_WORKERS

# --- Local SQLite-backed Iceberg sink config ---
LOCAL_WAREHOUSE = pathlib.Path(".load/orchestrate-local")
LOCAL_CATALOG_CONFIG = SqliteCatalogConfig(
    uri=f"sqlite:///{(LOCAL_WAREHOUSE / 'catalog.db').resolve()}",
    warehouse=str(LOCAL_WAREHOUSE.resolve()),
)


def _local_table_config_like(sink: IcebergTableSink) -> IcebergTableConfig:
    return IcebergTableConfig(
        catalog_config=LOCAL_CATALOG_CONFIG,
        namespace=sink.iceberg_table_config.namespace,
        table_name=sink.iceberg_table_config.table_name,
    )


_STRUCTURED_METADATA_TABLE_CONFIG = _local_table_config_like(
    runners_defaults.STRUCTURED_TABLE_SINK
)
STRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind=runners_defaults.STRUCTURED_TABLE_SINK.schema_kind,
    iceberg_table_config=_STRUCTURED_METADATA_TABLE_CONFIG,
    partition_column=runners_defaults.STRUCTURED_TABLE_SINK.partition_column,
)

_UNSTRUCTURED_METADATA_TABLE_CONFIG = _local_table_config_like(
    runners_defaults.UNSTRUCTURED_TABLE_SINK
)
UNSTRUCTURED_TABLE_SINK = IcebergTableSink(
    schema_kind=runners_defaults.UNSTRUCTURED_TABLE_SINK.schema_kind,
    iceberg_table_config=_UNSTRUCTURED_METADATA_TABLE_CONFIG,
    partition_column=runners_defaults.UNSTRUCTURED_TABLE_SINK.partition_column,
)

_DEAD_LETTER_TABLE_CONFIG = _local_table_config_like(
    runners_defaults.DEAD_LETTER_TABLE_SINK
)
DEAD_LETTER_TABLE_SINK = IcebergTableSink(
    schema_kind=runners_defaults.DEAD_LETTER_TABLE_SINK.schema_kind,
    iceberg_table_config=_DEAD_LETTER_TABLE_CONFIG,
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
