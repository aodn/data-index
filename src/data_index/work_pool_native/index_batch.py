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
    expected_identity = set(
        zip(batch_df["bucket"], batch_df["key"], batch_df["version_id"], strict=True)
    )
    handle_identity = [
        (
            handle.object_ref.bucket,
            handle.object_ref.key,
            handle.object_ref.version_id,
        )
        for handle in handles
    ]
    fetched_identity = set(handle_identity)
    missing_identity = sorted(expected_identity - fetched_identity)
    extra_identity = sorted(fetched_identity - expected_identity)
    duplicate_handle_identity = len(handle_identity) - len(fetched_identity)
    expected_count = len(expected_identity)
    coverage_pct = (
        (len(fetched_identity) / expected_count * 100) if expected_count else 100.0
    )

    # Summarise how many handles came from each backing implementation.
    handle_type_counts = Counter(type(handle).__name__ for handle in handles)
    handle_types_summary = ", ".join(
        f"{handle_type}:{count}"
        for handle_type, count in sorted(handle_type_counts.items())
    )

    logger.info(
        f"Batch handle summary: expected={expected_count} fetched={len(handles)} "
        f"unique={len(fetched_identity)} missing={len(missing_identity)} "
        f"extra={len(extra_identity)} duplicate_identities={duplicate_handle_identity} "
        f"coverage={coverage_pct:.2f}% handle_types=[{handle_types_summary or 'none'}]"
    )
    if missing_identity or extra_identity:
        logger.warning(
            f"Extract reconciliation mismatch: missing={len(missing_identity)} "
            f"extra={len(extra_identity)}"
        )

    # Always publish one compact summary artifact.
    prefect.artifacts.create_table_artifact(
        key="extract-handle-summary",
        table=[
            {
                "expected_identities": expected_count,
                "fetched_handles": len(handles),
                "unique_fetched_identities": len(fetched_identity),
                "missing_identities": len(missing_identity),
                "extra_identities": len(extra_identity),
                "duplicate_handle_identities": duplicate_handle_identity,
                "coverage_pct": round(coverage_pct, 2),
                "handle_types": handle_types_summary or "none",
            }
        ],
        description="Batch extract coverage and handle type mix",
    )

    # Publish bounded mismatch samples only when relevant.
    if missing_identity:
        missing_sample = missing_identity[:_ARTIFACT_SAMPLE_LIMIT]
        prefect.artifacts.create_table_artifact(
            key="extract-missing-handle-sample",
            table=[
                {
                    "bucket": bucket,
                    "key": key,
                    "version_id": version_id,
                }
                for bucket, key, version_id in missing_sample
            ],
            description=(
                f"Missing handles sample ({len(missing_identity)} total, "
                f"showing first {len(missing_sample)})"
            ),
        )

    if extra_identity:
        extra_sample = extra_identity[:_ARTIFACT_SAMPLE_LIMIT]
        prefect.artifacts.create_table_artifact(
            key="extract-extra-handle-sample",
            table=[
                {
                    "bucket": bucket,
                    "key": key,
                    "version_id": version_id,
                }
                for bucket, key, version_id in extra_sample
            ],
            description=(
                f"Extra handles sample ({len(extra_identity)} total, "
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
