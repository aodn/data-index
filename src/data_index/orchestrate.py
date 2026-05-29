from __future__ import annotations

import typing

import polars
import prefect
import prefect.cache_policies
import prefect.task_runners

from data_index.extract import extract
from data_index.load import load
from data_index.protocols import (
    BatchPartitioner,
    FileFetcher,
    InventorySource,
    MetadataExtractor,
    StructuredSink,
    UnstructuredMetadata,
    UnstructuredSink,
)
from data_index.transform import transform
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


@prefect.task(retries=2, cache_policy=prefect.cache_policies.NO_CACHE)
def _process_batch(
    batch_df: polars.DataFrame,
    fetcher: FileFetcher,
    extractor: MetadataExtractor,
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink,
    metadata_factory: typing.Callable[
        [str, dict], UnstructuredMetadata
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
        metadata_factory: callable(s3_uri, data) → UnstructuredMetadata
        transform_max_workers: max threads per batch in the transform step; None uses Python's default
    """
    logger = prefect.get_run_logger()
    if metadata_factory is None:
        metadata_factory = DiskCachedUnstructuredMetadata

    logger.info(f"Provisioning sinks: `{structured_sink}`, `{unstructured_sink}`")
    structured_sink.provision()
    unstructured_sink.provision()

    logger.info(f"Provisioning inventory: `{inventory_source}`")
    inventory = inventory_source.inventory()

    logger.info(f"Batch workers: `{partitioner}, `{fetcher}`, `{extractor}`")
    logger.info(f"Dispatching ({len(inventory)} files total)")
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

    for future in futures:
        future.result()

    logger.info("All batches complete")
