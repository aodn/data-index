from __future__ import annotations

import typing

import polars
import prefect
import prefect.task_runners

from data_index.protocols import (
    BatchPartitioner,
    FileFetcher,
    InventorySource,
    MetadataExtractor,
    StructuredSink,
    UnstructuredSink,
    UnstructuredMetadata,
)
from data_index.extract import extract
from data_index.transform import transform
from data_index.load import load
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


@prefect.task(retries=2)
def _process_batch(
    batch_df: polars.DataFrame,
    fetcher: FileFetcher,
    extractor: MetadataExtractor,
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink,
    metadata_factory: typing.Callable[[str, dict], UnstructuredMetadata] = DiskCachedUnstructuredMetadata,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""
    handles = extract(batch_df=batch_df, fetcher=fetcher)
    results = transform(xarray_handles=handles, extractor=extractor, metadata_factory=metadata_factory)
    load(extraction_results=results, structured_sink=structured_sink, unstructured_sink=unstructured_sink)


@prefect.flow(task_runner=prefect.task_runners.ThreadPoolTaskRunner())
def orchestrate(
    inventory_source,
    partitioner,
    fetcher,
    extractor,
    structured_sink,
    unstructured_sink,
    metadata_factory=None,
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
    """
    logger = prefect.get_run_logger()
    if metadata_factory is None:
        metadata_factory = DiskCachedUnstructuredMetadata

    inventory = inventory_source.inventory()
    batches = list(partitioner.partition(inventory))
    logger.info(f"Dispatching {len(batches)} batches ({len(inventory)} files total)")

    futures = [
        _process_batch.submit(
            batch_df=batch,
            fetcher=fetcher,
            extractor=extractor,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
            metadata_factory=metadata_factory,
        )
        for batch in batches
    ]

    for future in futures:
        future.result()

    logger.info("All batches complete")
