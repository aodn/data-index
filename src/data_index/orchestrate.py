from __future__ import annotations

import typing

import polars
import prefect
import prefect.cache_policies
import prefect.futures
import prefect.task_runners

from data_index.extract import extract
from data_index.load import load
from data_index.protocols import (
    BatchPartitioner,
    FileFetcher,
    InventorySource,
    MetadataExtractor,
    ObjectReference,
    StructuredSink,
    UnstructuredMetadata,
    UnstructuredSink,
)
from data_index.transform import transform
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


@prefect.task
def _warm_task_runner():
    """
    Essentially does nothing, but allows a dask cluster to provision
    in the background if necesssary.
    """
    logger = prefect.get_run_logger()
    runner = prefect.context.FlowRunContext.get().task_runner
    logger.info(f"Warming up task runner ({type(runner)})...")


@prefect.task(retries=2, cache_policy=prefect.cache_policies.NO_CACHE)
def _process_batch(
    batch_df: polars.DataFrame,
    fetcher: FileFetcher,
    extractor: MetadataExtractor,
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink,
    metadata_factory: typing.Callable[
        [ObjectReference, dict], UnstructuredMetadata
    ] = DiskCachedUnstructuredMetadata,
    transform_max_workers: int | None = None,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""
    handles = extract(batch_df=batch_df, fetcher=fetcher)
    results = transform(
        xarray_handles=handles,
        extractor=extractor,
        metadata_factory=metadata_factory,
        max_workers=transform_max_workers,
    )
    load(
        extraction_results=results,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
    )


@prefect.flow(task_runner=prefect.task_runners.ThreadPoolTaskRunner())
def orchestrate(
    inventory_source: InventorySource,
    partitioner: BatchPartitioner,
    fetcher: FileFetcher,
    extractor: MetadataExtractor,
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink,
    metadata_factory=None,
    transform_max_workers: int | None = None,
) -> None:
    """Orchestrator flow: read inventory → partition → dispatch ETL tasks as concurrent workers.

    Defaults to ThreadPoolTaskRunner for local runs. For Fargate dispatch use:
        orchestrate.with_options(task_runner=DaskTaskRunner(...))(...)

    Sinks and other dependencies are injected so the flow is testable without live infrastructure.

    Args:
        inventory_source: InventorySource — provides the full corpus inventory DataFrame
        partitioner: BatchPartitioner — splits inventory into Batches
        fetcher: FileFetcher — fetches files for each Batch
        extractor: MetadataExtractor — extracts structured/unstructured metadata
        structured_sink: StructuredSink — persists structured metadata
        unstructured_sink: UnstructuredSink — persists unstructured metadata
        metadata_factory: callable(object_ref, data) → UnstructuredMetadata
        transform_max_workers: max threads per batch in the transform step; None uses Python's default
    """

    logger = prefect.get_run_logger()

    # Warm up the task runner
    future = [_warm_task_runner.submit()]

    # Set the unstructured metadata factory
    # This is in place to allow switching from
    # in memory json to on disk json...
    # Unstructured metadata can be quite large
    if metadata_factory is None:
        metadata_factory = DiskCachedUnstructuredMetadata

    # Provision the inventory (files to index)
    logger.info(f"Provisioning inventory: `{inventory_source}`")
    inventory = inventory_source.inventory()

    # Provision the sinks
    logger.info(f"Provisioning sinks: `{structured_sink}`, `{unstructured_sink}`")
    structured_sink.provision()
    unstructured_sink.provision()

    # Wait for task runner
    prefect.futures.wait(future)
    logger.info("Cluster is warm. Dispatching ETL batch workers...")

    # Dispatch
    # Note: Scheduler has to be able to hold all the dispatch
    # data; that includes all s3 keys.
    logger.info(f"Dispatching ({len(inventory)} files total)")
    logger.info(f"Batch workers: `{partitioner}, `{fetcher}`, `{extractor}`")
    futures = [
        _process_batch.submit(
            batch_df=batch,
            fetcher=fetcher,
            extractor=extractor,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
            metadata_factory=metadata_factory,
            transform_max_workers=transform_max_workers,
        )
        for batch in partitioner.partition(inventory)
    ]

    # Stream results
    logger.info(f"All {len(futures)} batches dispatched. Streaming results...")
    for future in prefect.futures.as_completed(futures=futures):
        logger.info(future.state)
        logger.info(future.result(raise_on_failure=False))
    logger.info("All batches complete")
