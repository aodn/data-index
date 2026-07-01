"""
Data Indexing and Orchestration Pipeline Module.

This module provides a scalable, distributed orchestration architecture using Prefect
to discover, partition, fetch, extract, and sink metadata from large datasets. It is
designed to handle high-throughput file inventories (such as NetCDF repositories stored
on S3) by decomposing the processing load into concurrent, independent batch deployments.

Architecture Overview
---------------------
The orchestration is organized into three distinct operational layers:

1. **Top-Level Orchestration Flow (`index`):** Acts as the user-facing entrypoint. It configures the runtime environment, injecting
   concrete infrastructure choices (such as specific S3 sources, NetCDF extractors, and
   process/thread-pool task runners) before delegating to the core execution logic.

2. **Core Pipeline Engine (`index_pipeline`):** Manages the data pipeline lifecycle. It provisions the structured and unstructured
   target sinks, extracts the global file inventory, partitions the workload using a
   pluggable strategy, and submits chunks concurrently as managed sub-tasks.

3. **Distributed Batch Worker Task (`index_batch`):** A proxy task that offloads processing execution by triggering independent Prefect
   deployments (`process-batch-{i}`) over the wire, protecting the parent scheduler's
   memory footprint and isolating batch failures.

Design & Constraints
--------------------
* **Extensibility:** Built around abstract base components (:class:`InventorySource`,
  :class:`BatchPartitioner`, :class:`FileFetcher`, :class:`MetadataExtractor`, and
  :class:`Sink` classes), allowing the pipeline to adapt to shifting underlying file
  formats or network protocol topologies.
"""

import pathlib
import typing

import prefect
import prefect.deployments
import prefect.futures
import prefect.states

import data_index.protocols
from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import (
    ConcurrentObstoreFetcher,
    FSSpecFetcher,
    ObstoreFetcher,
)
from data_index.iceberg_config import (
    IcebergTableConfig,
    IcebergTableScanConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import (
    DeltaIcebergTableInventorySource,
    IcebergTableInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.runners.task_runner import (
    ProcessPoolRunnerConfig,
    ThreadPoolRunnerConfig,
)
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata
from data_index.sink import (
    DummySink,
    IcebergTableSink,
)

InventorySource: typing.TypeAlias = (
    DeltaIcebergTableInventorySource | IcebergTableInventorySource
)
BatchPartitioner: typing.TypeAlias = GreedyBatchPartitioner
FileFetcher: typing.TypeAlias = (
    FSSpecFetcher | ObstoreFetcher | ConcurrentObstoreFetcher
)
MetadataExtractor: typing.TypeAlias = AttributeNetCDFExtractor
MetadataSink: typing.TypeAlias = IcebergTableSink | DummySink

# --- General config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
BATCH_SIZE = 1_000
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2 * 10  # 100 MB
S5CMD_WORKERS = 8


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

_inventory_source = IcebergTableInventorySource(
    table_config=_inventory_table_config,
    table_scan_config=IcebergTableScanConfig(
        row_filter="key LIKE 'IMOS/SOOP/%' OR key LIKE 'IMOS/AATAMS/%' OR key LIKE 'IMOS/ANMN/%' OR key LIKE 'IMOS/FAIMMS/%' OR key LIKE 'IMOS/OceanCurrent/%' OR key LIKE 'IMOS/DWM/%' OR key LIKE 'IMOS/AUV/%' OR key LIKE 'IMOS/COASTAL-WAVE-BUOYS/%' OR key LIKE 'IMOS/NTP/%' OR key LIKE 'IMOS/ANFOG/%' OR key LIKE 'IMOS/eMII/%'",
        selected_fields=["bucket", "key", "version_id", "size"],
    ),
)

# --- Partitioner config ---
_greedy_partitioner = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=10 * 1024**3,
)

# --- File fetcher ---
_file_fetcher = ObstoreFetcher()

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

_structured_table_sink = IcebergTableSink(
    schema_kind="structured",
    iceberg_table_config=_structured_metadata_table_config,
    partition_column="facility",
)

_unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name=f"unstructured_metadata_v{UnstructuredMetadata.SCHEMA_VERSION}",
)

_unstructured_table_sink = IcebergTableSink(
    schema_kind="unstructured",
    iceberg_table_config=_unstructured_metadata_table_config,
    partition_column="facility",
)

_dead_letter_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name=f"dead_letter_v{data_index.protocols.DeadLetter.SCHEMA_VERSION}",
)

_dead_letter_table_sink = IcebergTableSink(
    schema_kind="dead_letter",
    iceberg_table_config=_dead_letter_table_config,
)

# --- Runtime Config ---
_task_runner_config = ThreadPoolRunnerConfig()


def _split_object_reference_batch(
    object_reference_batch: list[data_index.protocols.ObjectReference],
    max_size_bytes: int = 512 * 1024,
) -> list[str]:
    """
    Recursively splits a list of data_index.protocols.ObjectReferences until each batch
    compressed base64 string is under the specified byte limit.
    """
    compressed_batch = data_index.protocols.ObjectReference.to_compressed_base64_table(
        object_references=object_reference_batch
    )

    # Check size
    if len(compressed_batch.encode("utf-8")) <= max_size_bytes:
        return [compressed_batch]

    # If a single item is already over the limit, we cannot split further
    if len(object_reference_batch) <= 1:
        raise ValueError("Single object reference exceeds the 512KB limit.")

    # Split the list in half
    mid = len(object_reference_batch) // 2
    left_batch = object_reference_batch[:mid]
    right_batch = object_reference_batch[mid:]

    # Recurse
    return _split_object_reference_batch(
        left_batch, max_size_bytes
    ) + _split_object_reference_batch(right_batch, max_size_bytes)


@prefect.task(
    task_run_name="index-batch-{i}",
)
def index_batch(
    i: int,
    index_batch_flow_name: str,
    index_batch_deployment_name: str,
    object_reference_batch: list[data_index.protocols.ObjectReference],
    fetcher: data_index.protocols.FileFetcher,
    extractor: data_index.protocols.MetadataExtractor,
    structured_sink: data_index.protocols.MetadataSink,
    unstructured_sink: data_index.protocols.MetadataSink,
    dead_letter_sink: data_index.protocols.MetadataSink,
    max_workers: int | None = 8,
):
    """
    Submit and monitor a specific sub-batch indexing deployment run.

    This Prefect task acts as a proxy that triggers a remote deployment run for a specific
    batch of object references. It passes the necessary processing tools (fetcher, extractor,
    and sinks) as parameters, monitors the resulting flow run state, and handles downstream
    failures or anomalous uncompleted states.

    :param i: The index identifier for the specific batch run.
    :type i: int
    :param index_batch_flow_name: Name of the target batch processing flow.
    :type index_batch_flow_name: str
    :param index_batch_deployment_name: Name of the deployment associated with the batch flow.
    :type index_batch_deployment_name: str
    :param object_reference_batch: A collection of data objects to be processed within this batch.
    :type object_reference_batch: list[data_index.protocols.ObjectReference]
    :param fetcher: File transfer interface to pull files for processing.
    :type fetcher: FileFetcher
    :param extractor: Interface used to pull targeted metadata out of fetched files.
    :type extractor: MetadataExtractor
    :param structured_sink: Storage destination interface for structured data targets.
    :type structured_sink: StructuredSink
    :param unstructured_sink: Storage destination interface for unstructured data targets.
    :type unstructured_sink: UnstructuredSink

    :return: None
    :rtype: None

    :raises RuntimeError: If the triggered deployment run finalizes with an unidentifiable state (None).
    :raises Exception: Re-raises any execution exception caught via Prefect's state tracking if the sub-flow fails.
    """

    # Compress the object reference batch
    compressed_object_reference_batch = (
        data_index.protocols.ObjectReference.to_compressed_base64_table(
            object_references=object_reference_batch
        )
    )

    # Check compression succeeded to the degree necessary
    max_bytes = 500 * 1024
    if len(compressed_object_reference_batch) > 500 * 1024:
        raise RuntimeError(
            f"compression of data_index.protocols.ObjectReferences failed to fall under target size; required <= {max_bytes} and got {len(compressed_object_reference_batch)}"
        )

    # Run the index batch
    flow_run = prefect.deployments.run_deployment(
        name=f"{index_batch_flow_name}/{index_batch_deployment_name}",
        flow_run_name=f"process-batch-{i}",
        parameters={
            "compressed_object_reference_batch": compressed_object_reference_batch,
            "fetcher": fetcher,
            "extractor": extractor,
            "structured_sink": structured_sink,
            "unstructured_sink": unstructured_sink,
            "dead_letter_sink": dead_letter_sink,
            "max_workers": max_workers,
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


@prefect.flow
def index_pipeline(
    inventory_source: data_index.protocols.InventorySource,
    partitioner: data_index.protocols.BatchPartitioner,
    fetcher: data_index.protocols.FileFetcher,
    extractor: data_index.protocols.MetadataExtractor,
    structured_sink: data_index.protocols.MetadataSink,
    unstructured_sink: data_index.protocols.MetadataSink,
    dead_letter_sink: data_index.protocols.MetadataSink,
    index_batch_flow_name: str = "index-batch",
    index_batch_deployment_name: str = "index-batch",
    batch_max_workers: int | None = None,
):
    """
    Execute the core data indexing pipeline by batching and dispatching file inventory.

    This flow handles the lifecycle of the indexing operation: extracting the inventory,
    provisioning storage sinks, partitioning the dataset, concurrently submitting batch
    indexing tasks via Prefect futures, and streaming the execution states to catch errors.

    .. note::
        The scheduler acts as the central hub holding all dispatch data (including
        S3 keys and version IDs). This can result in a significantly large memory
        footprint for vast inventories.

    :param inventory_source: Component responsible for extracting the file inventory list.
    :type inventory_source: InventorySource
    :param partitioner: Strategy to segment the core inventory into manageable batches.
    :type partitioner: BatchPartitioner
    :param fetcher: File transfer interface to pull files for processing.
    :type fetcher: FileFetcher
    :param extractor: Interface used to pull targeted metadata out of fetched files.
    :type extractor: MetadataExtractor
    :param structured_sink: Storage destination interface for structured data targets.
    :type structured_sink: StructuredSink
    :param unstructured_sink: Storage destination interface for unstructured data targets.
    :type unstructured_sink: UnstructuredSink
    :param index_batch_flow_name: Name of the target batch sub-flow, defaults to "index-batch".
    :type index_batch_flow_name: str, optional
    :param index_batch_deployment_name: Name of the target batch deployment configuration, defaults to "index-batch".
    :type index_batch_deployment_name: str, optional

    :raises RuntimeError: If one or more submitted batch indexing tasks fail during execution.
    """
    logger = prefect.get_run_logger()

    # Provision the inventory (files to index)
    logger.info(f"Extracting inventory: `{inventory_source}`")
    inventory = inventory_source.inventory()

    # Provision the sinks
    logger.info(f"Provisioning structured sink: {structured_sink}")
    structured_sink.provision()
    logger.info(f"Provisioning unstructured sink: {unstructured_sink}")
    unstructured_sink.provision()
    logger.info(f"Provisioning dead letter sink: {dead_letter_sink}")
    dead_letter_sink.provision()

    # Dispatch
    # Note: Scheduler has to be able to hold all the dispatch
    # data; that includes all S3 keys and version_ids, so could have a large memory footprint
    logger.info(f"Dispatching ({len(inventory)} files total)")
    logger.info(f"Batch workers: `{partitioner}, `{fetcher}`, `{extractor}`")

    object_reference_batch_generator = partitioner.partition(inventory=inventory)

    futures = [
        index_batch.submit(
            i=i,
            index_batch_flow_name=index_batch_flow_name,
            index_batch_deployment_name=index_batch_deployment_name,
            object_reference_batch=object_reference_batch,
            fetcher=fetcher,
            extractor=extractor,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
            dead_letter_sink=dead_letter_sink,
            max_workers=batch_max_workers,
        )
        for i, object_reference_batch in enumerate(object_reference_batch_generator)
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


@prefect.flow
def index(
    inventory_source: InventorySource = _inventory_source,
    partitioner: BatchPartitioner = _greedy_partitioner,
    fetcher: FileFetcher = _file_fetcher,
    extractor: MetadataExtractor = _attribute_netcdf_extractor,
    structured_sink: MetadataSink = _structured_table_sink,
    unstructured_sink: MetadataSink = _unstructured_table_sink,
    dead_letter_sink: MetadataSink = _dead_letter_table_sink,
    index_batch_flow_name: str = "index-batch",
    index_batch_deployment_name: str = "index-batch",
    task_runner_config: ProcessPoolRunnerConfig
    | ThreadPoolRunnerConfig = _task_runner_config,
    batch_max_workers: int | None = None,
):
    """
    Orchestrate the end-to-end data indexing pipeline.

    This Prefect flow coordinates the extraction, partitioning, fetching,
    and sinking of dataset attributes (NetCDF) into structured and
    unstructured S3 tables using a configurable concurrent task runner.

    :param inventory_source: The source containing table inventory metadata.
    :type inventory_source: S3TableInventorySource | S3TableFacilitySubsetInventorySource
    :param partitioner: Strategy used to partition dataset batches.
    :type partitioner: GreedyBatchPartitioner
    :param fetcher: The file transfer client used to fetch remote files.
    :type fetcher: FSSpecFetcher | ObstoreFetcher
    :param extractor: The metadata extractor tailored for NetCDF attributes.
    :type extractor: AttributeNetCDFExtractor
    :param structured_sink: Target destination for structured data output.
    :type structured_sink: StructuredS3TableSink
    :param unstructured_sink: Target destination for unstructured data output.
    :type unstructured_sink: UnstructuredS3TableSink
    :param index_batch_flow_name: Name of the downstream batch flow to invoke, defaults to "index-batch".
    :type index_batch_flow_name: str, optional
    :param index_batch_deployment_name: Name of the downstream batch deployment, defaults to "index-batch".
    :type index_batch_deployment_name: str, optional
    :param task_runner_config: Configuration for the concurrent task runner infrastructure.
    :type task_runner_config: ProcessPoolRunnerConfig | ThreadPoolRunnerConfig

    :raises PrefectException: If any upstream task fails or the sub-pipeline execution encounters an error.
    """

    # Run the index pipeline with depenencies
    logger = prefect.get_run_logger()
    logger.info(f"Executing pipeline with task runner: `{task_runner_config}`...")
    (
        index_pipeline.with_options(
            task_runner=task_runner_config.create(),
        )(
            inventory_source=inventory_source,
            partitioner=partitioner,
            fetcher=fetcher,
            extractor=extractor,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
            dead_letter_sink=dead_letter_sink,
            index_batch_flow_name=index_batch_flow_name,
            index_batch_deployment_name=index_batch_deployment_name,
            batch_max_workers=batch_max_workers,
        )
    )
    logger.info("Executed pipeline!")


if __name__ == "__main__":
    index.serve(
        name="index",
    )
