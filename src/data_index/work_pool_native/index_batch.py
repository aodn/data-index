import polars
import prefect
import prefect.artifacts
import prefect.cache_policies

from data_index.extract import extract
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.load import load
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
    NetCDFExtractor,
    UnstructuedNetCDFExtractor,
)
from data_index.structured_sink import StructuredParquetSink, StructuredS3TableSink
from data_index.transform import transform
from data_index.unstructured_sink import (
    UnstructuredParquetSink,
    UnstructuredS3TableSink,
)


@prefect.flow
def index_batch(
    batch,
    fetcher: S3Fetcher | S5CMDFetcher | ThresholdFileFetcher,
    extractor: AttributeNetCDFExtractor | NetCDFExtractor | UnstructuedNetCDFExtractor,
    structured_sink: StructuredParquetSink | StructuredS3TableSink,
    unstructured_sink: UnstructuredParquetSink | UnstructuredS3TableSink,
    transform_max_workers: int,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""

    logger = prefect.get_run_logger()

    # De-serialize batch
    logger.info("De-serializing batch...")
    batch_df = polars.DataFrame(data=batch)
    logger.info(f"De-serialized batch! Got `{len(batch_df)}` rows")

    # Extract batch
    logger.info("Extracting batch...")
    handles = extract(batch_df=batch_df, fetcher=fetcher)
    logger.info("Extracted batch!")

    # Transform batch
    logger.info("Transforming batch...")
    results = transform(
        xarray_handles=handles,
        extractor=extractor,
        max_workers=transform_max_workers,
    )
    logger.info("Transformed batch!")

    # Load batch
    logger.info("Loading batch...")
    load(
        extraction_results=results,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
    )
    logger.info("Loaded batch!")


if __name__ == "__main__":
    index_batch.serve(
        name="index-batch",
        global_limit=8,
    )
