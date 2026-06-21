import pathlib

import prefect
import prefect.deployments
import prefect.futures
import prefect.states
import prefect.task_runners

from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config import (
    IcebergTableConfig,
    IcebergTableScanConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import (
    S3TableFacilitySubsetInventorySource,
    S3TableInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
    NetCDFExtractor,
    UnstructuedNetCDFExtractor,
)
from data_index.structured_metadata import StructuredMetadata
from data_index.structured_sink import StructuredParquetSink, StructuredS3TableSink
from data_index.unstructured_sink import (
    UnstructuredParquetSink,
    UnstructuredS3TableSink,
)

# --- General config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
BATCH_SIZE = 1_000
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB
S5CMD_WORKERS = 8
TRANSFORM_MAX_WORKERS = 4


# --- Live Inventory Source config
_s3_metadata_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
)

_inventory_table_config = IcebergTableConfig(
    catalog_config=_s3_metadata_catalog_config,
    namespace="inventory",
    table_name="live",
)

_inventory_source = S3TableInventorySource(
    table_config=_inventory_table_config,
    table_scan_config=IcebergTableScanConfig(
        row_filter="""
        (
            key LIKE 'IMOS/ANMN/AM/%'
            OR key LIKE 'IMOS/ANMN/NRS/%'
            OR key LIKE 'IMOS/ANMN/NSW/%'
            OR key LIKE 'IMOS/ANMN/QLD/%'
            OR key LIKE 'IMOS/ANMN/SA/%'
            OR key LIKE 'IMOS/ANMN/WA/%'
        )
        AND NOT key LIKE 'IMOS/ANMN/NRS/REAL_TIME/%'
        """
    ),
)

# --- Partitioner config ---
_greedy_partitioner = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=10 * 1024**3,
)

# --- File fetcher ---
_file_fetcher = S5CMDFetcher(num_workers=S5CMD_WORKERS)

# --- Metadata extractor ---
_attribute_netcdf_extractor = AttributeNetCDFExtractor()

# --- Sink config ---
_data_index_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)

_structured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

_structured_s3_table_sink = StructuredS3TableSink(
    iceberg_table_config=_structured_metadata_table_config,
)

_unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name="unstructured_metadata",
)

_unstructured_s3_table_sink = UnstructuredS3TableSink(
    iceberg_table_config=_unstructured_metadata_table_config,
)


@prefect.task(
    task_run_name="index-batch-{i}",
)
def index_batch(
    i,
    index_batch_flow_name,
    index_batch_deployment_name,
    batch_df,
    fetcher,
    extractor,
    structured_sink,
    unstructured_sink,
    transform_max_workers: int,
):

    # Run the index batch
    flow_run = prefect.deployments.run_deployment(
        name=f"{index_batch_flow_name}/{index_batch_deployment_name}",
        flow_run_name=f"process-batch-{i}",
        parameters={
            "batch": batch_df.to_dicts(),
            "fetcher": fetcher,
            "extractor": extractor,
            "structured_sink": structured_sink,
            "unstructured_sink": unstructured_sink,
            "transform_max_workers": transform_max_workers,
        },
    )

    # Raise unknown state error
    if flow_run.state is None:
        raise RuntimeError(
            f"Flow run process-batch-{i} finalised with unknown state (`flow_run.state` == None)!"
        )

    # Raise the exception if it failed
    prefect.states.raise_state_exception(flow_run.state)
    return


@prefect.flow(task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=8))
def run_index_work_pool(
    inventory_source: S3TableInventorySource
    | S3TableFacilitySubsetInventorySource = _inventory_source,
    partitioner: GreedyBatchPartitioner = _greedy_partitioner,
    fetcher: S3Fetcher | S5CMDFetcher | ThresholdFileFetcher = _file_fetcher,
    extractor: AttributeNetCDFExtractor
    | NetCDFExtractor
    | UnstructuedNetCDFExtractor = _attribute_netcdf_extractor,
    structured_sink: StructuredParquetSink
    | StructuredS3TableSink = _structured_s3_table_sink,
    unstructured_sink: UnstructuredParquetSink
    | UnstructuredS3TableSink = _unstructured_s3_table_sink,
    index_batch_flow_name="index-batch",
    index_batch_deployment_name="index-batch",
    transform_max_workers: int = TRANSFORM_MAX_WORKERS,
):

    logger = prefect.get_run_logger()

    # Provision the inventory (files to index)
    logger.info(f"Extracting inventory: `{inventory_source}`")
    inventory = inventory_source.inventory()

    # Provision the sinks
    logger.info(f"Provisioning sinks: `{structured_sink}`, `{unstructured_sink}`")
    structured_sink.provision()
    unstructured_sink.provision()

    # Dispatch
    # Note: Scheduler has to be able to hold all the dispatch
    # data; that includes all s3 keys.
    logger.info(f"Dispatching ({len(inventory)} files total)")
    logger.info(f"Batch workers: `{partitioner}, `{fetcher}`, `{extractor}`")

    futures = [
        index_batch.submit(
            i=i,
            index_batch_flow_name=index_batch_flow_name,
            index_batch_deployment_name=index_batch_deployment_name,
            batch_df=batch_df,
            fetcher=fetcher,
            extractor=extractor,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
            transform_max_workers=transform_max_workers,
        )
        for i, batch_df in enumerate(
            iterable=partitioner.partition(inventory=inventory),
        )
    ]

    #  Stream results
    logger.info(
        f"All {len(futures)} batches internally scheduled... Streaming results..."
    )
    failed = []
    for future in prefect.futures.as_completed(futures=futures):
        logger.info(future.state)
        try:
            prefect.states.raise_state_exception(future.state)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            failed.append(e)

    # Report overall status
    if failed:
        raise RuntimeError(f"{len(failed)} batch(es) failed. See logs for details.")
    else:
        logger.info("All batches completed successfully!")


if __name__ == "__main__":
    run_index_work_pool.serve(
        name="run-index-work-pool",
    )
