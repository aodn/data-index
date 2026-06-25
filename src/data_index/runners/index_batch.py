import prefect

import data_index
from data_index.file_fetcher import FSSpecFetcher, ObstoreFetcher
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.protocols import ObjectReference
from data_index.sink import IcebergTableSink


@prefect.flow
def index_batch(
    object_reference_batch: list[ObjectReference],
    fetcher: FSSpecFetcher | ObstoreFetcher,
    extractor: AttributeNetCDFExtractor,
    structured_sink: IcebergTableSink,
    unstructured_sink: IcebergTableSink,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""

    logger = prefect.get_run_logger()

    # Extract batch
    logger.info("Extracting batch...")
    staged_objects, dead_letters = data_index.extract(
        object_references=object_reference_batch,
        fetcher=fetcher,
    )
    logger.info("Extracted batch!")

    # TODO: Deal with dead letters

    # Transform batch
    logger.info("Transforming batch...")
    extracted_objects, dead_letters = data_index.transform(
        staged_objects=staged_objects,
        extractor=extractor,
    )
    logger.info("Transformed batch!")

    # TODO: Deal with dead letters

    # Load batch
    logger.info("Loading batch...")
    data_index.load(
        extracted_objects=extracted_objects,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
    )
    logger.info("Loaded batch!")


if __name__ == "__main__":
    index_batch.serve(
        name="index-batch",
        global_limit=12,
    )
