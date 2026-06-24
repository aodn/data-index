import prefect

from data_index.extract import extract
from data_index.file_fetcher import FSSpecFetcher, ObstoreFetcher
from data_index.load import load
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.protocols import ObjectReference
from data_index.structured_sink import StructuredS3TableSink
from data_index.transform import transform
from data_index.unstructured_sink import (
    UnstructuredS3TableSink,
)


@prefect.flow
def index_batch(
    object_reference_batch: list[ObjectReference],
    fetcher: FSSpecFetcher | ObstoreFetcher,
    extractor: AttributeNetCDFExtractor,
    structured_sink: StructuredS3TableSink,
    unstructured_sink: UnstructuredS3TableSink,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""

    logger = prefect.get_run_logger()

    return

    # Extract batch
    logger.info("Extracting batch...")
    object_references = extract(
        object_references=object_reference_batch,
        fetcher=fetcher,
    )
    logger.info("Extracted batch!")

    # Transform batch
    logger.info("Transforming batch...")
    extraction_results = transform(
        object_references=object_references,
        extractor=extractor,
    )
    logger.info("Transformed batch!")

    # Load batch
    logger.info("Loading batch...")
    load(
        extraction_results=extraction_results,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
    )
    logger.info("Loaded batch!")


if __name__ == "__main__":
    index_batch.serve(
        name="index-batch",
        global_limit=12,
    )
