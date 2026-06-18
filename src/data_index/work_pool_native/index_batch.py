from collections import Counter

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

_ARTIFACT_SAMPLE_LIMIT = 25


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE, retries=0)
def _summarise_batch_handles(
    batch_df: polars.DataFrame,
    handles: list,
) -> None:
    """Summarise extract output fidelity and handle mix with bounded artifact volume.

    Reconciles expected batch URIs against returned handles, logs a compact
    coverage summary, and publishes small Prefect artifacts for diagnostics.
    Any mismatch details are capped to a fixed sample size to avoid large
    artifact payloads on big batches.
    """

    logger = prefect.get_run_logger()

    # Reconcile what we expected to fetch with what extract returned.
    expected_uris = set(batch_df["s3_uri"].to_list())
    handle_uris = [handle.s3_uri for handle in handles]
    fetched_uris = set(handle_uris)
    missing_uris = sorted(expected_uris - fetched_uris)
    extra_uris = sorted(fetched_uris - expected_uris)
    duplicate_handle_uris = len(handle_uris) - len(fetched_uris)
    expected_count = len(expected_uris)
    coverage_pct = (
        (len(fetched_uris) / expected_count * 100) if expected_count else 100.0
    )

    # Summarise how many handles came from each backing implementation.
    handle_type_counts = Counter(type(handle).__name__ for handle in handles)
    handle_types_summary = ", ".join(
        f"{handle_type}:{count}"
        for handle_type, count in sorted(handle_type_counts.items())
    )

    logger.info(
        f"Batch handle summary: expected={expected_count} fetched={len(handles)} "
        f"unique={len(fetched_uris)} missing={len(missing_uris)} "
        f"extra={len(extra_uris)} duplicate_uris={duplicate_handle_uris} "
        f"coverage={coverage_pct:.2f}% handle_types=[{handle_types_summary or 'none'}]"
    )
    if missing_uris or extra_uris:
        logger.warning(
            f"Extract reconciliation mismatch: missing={len(missing_uris)} "
            f"extra={len(extra_uris)}"
        )

    # Always publish one compact summary artifact.
    prefect.artifacts.create_table_artifact(
        key="extract-handle-summary",
        table=[
            {
                "expected_uris": expected_count,
                "fetched_handles": len(handles),
                "unique_fetched_uris": len(fetched_uris),
                "missing_uris": len(missing_uris),
                "extra_uris": len(extra_uris),
                "duplicate_handle_uris": duplicate_handle_uris,
                "coverage_pct": round(coverage_pct, 2),
                "handle_types": handle_types_summary or "none",
            }
        ],
        description="Batch extract coverage and handle type mix",
    )

    # Publish bounded mismatch samples only when relevant.
    if missing_uris:
        missing_sample = missing_uris[:_ARTIFACT_SAMPLE_LIMIT]
        prefect.artifacts.create_table_artifact(
            key="extract-missing-handle-sample",
            table=[{"s3_uri": uri} for uri in missing_sample],
            description=(
                f"Missing handles sample ({len(missing_uris)} total, "
                f"showing first {len(missing_sample)})"
            ),
        )

    if extra_uris:
        extra_sample = extra_uris[:_ARTIFACT_SAMPLE_LIMIT]
        prefect.artifacts.create_table_artifact(
            key="extract-extra-handle-sample",
            table=[{"s3_uri": uri} for uri in extra_sample],
            description=(
                f"Extra handles sample ({len(extra_uris)} total, "
                f"showing first {len(extra_sample)})"
            ),
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
    _summarise_batch_handles(
        batch_df=batch_df,
        handles=handles,
    )

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
