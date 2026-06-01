from __future__ import annotations

import concurrent.futures
import logging
import typing

import polars
import prefect
import prefect.artifacts
import prefect.cache_policies

from data_index.protocols import (
    ExtractionResult,
    MetadataExtractor,
    UnstructuredMetadata,
    XarrayHandle,
)
from data_index.structured_metadata import StructuredMetadata
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


def _transform_single(
    xarray_handle: XarrayHandle,
    extractor: MetadataExtractor,
    unstructured_metadata_factory: typing.Callable[[str, dict], UnstructuredMetadata],
    logger: logging.Logger,
) -> ExtractionResult:
    try:
        raw = extractor.extract(handle=xarray_handle)
        if raw.status == "failed":
            logger.warning(f"extraction failed for {xarray_handle.s3_uri}: {raw.error}")
            return ExtractionResult(
                s3_uri=xarray_handle.s3_uri,
                structured_metadata=None,
                unstructured_metadata=None,
                status="failed",
                error=raw.error,
            )
        logger.info(f"extraction succeeded for {xarray_handle.s3_uri}")
        return ExtractionResult(
            s3_uri=xarray_handle.s3_uri,
            structured_metadata=raw.structured_metadata,
            unstructured_metadata=unstructured_metadata_factory(
                xarray_handle.s3_uri, raw.unstructured_metadata
            ),
            status="succeeded",
        )
    except Exception as exc:
        logger.warning(f"extraction failed for {xarray_handle.s3_uri}: {exc}")
        return ExtractionResult(
            s3_uri=xarray_handle.s3_uri,
            structured_metadata=None,
            unstructured_metadata=None,
            status="failed",
            error=str(exc),
        )
    finally:
        xarray_handle.ds.close()


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE)
def transform(
    xarray_handles: list[XarrayHandle],
    extractor: MetadataExtractor,
    metadata_factory: typing.Callable[
        [str, dict], UnstructuredMetadata
    ] = DiskCachedUnstructuredMetadata,
    max_workers: int | None = None,
) -> list[ExtractionResult]:
    """
    Transform a list of XarrayHandle objects into structured and unstructured metadata.

    Runs one _transform_single call per file in parallel via a thread pool. Each call
    immediately persists unstructured metadata via metadata_factory(s3_uri, data).
    Releases handle resources after all threads complete.

    Args:
        max_workers: Maximum threads for the internal pool. None uses Python's default
            (min(32, cpu_count + 4)). Set explicitly when multiple batches run
            concurrently to avoid thread explosion across workers.

    Returns list of ExtractionResult (succeeded and failed). Callers route to sinks.
    """
    logger = prefect.get_run_logger()
    total = len(xarray_handles)
    progress_artifact_id = prefect.artifacts.create_progress_artifact(
        progress=0.0,
        description=f"Transforming {total} files",
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _transform_single,
                xarray_handle=xarray_handle,
                extractor=extractor,
                unstructured_metadata_factory=metadata_factory,
                logger=logger,
            )
            for xarray_handle in xarray_handles
        }
        results = []
        for done_count, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            results.append(future.result())
            prefect.artifacts.update_progress_artifact(
                artifact_id=progress_artifact_id,
                progress=done_count / total * 100 if total else 100.0,
            )

    # Release handle resources (e.g. delete local files for DiskXarrayHandle)
    for xarray_handle in xarray_handles:
        xarray_handle.cleanup()

    failed = [r for r in results if r.status == "failed"]
    succeeded = [r for r in results if r.status == "succeeded"]

    prefect.artifacts.create_table_artifact(
        key="transform-succeeded",
        table=[{"s3_uri": r.s3_uri} for r in succeeded],
        description=f"{len(succeeded)}/{len(results)} files succeeded",
    )
    prefect.artifacts.create_table_artifact(
        key="transform-failed",
        table=[{"s3_uri": r.s3_uri, "error": r.error} for r in failed]
        or [{"s3_uri": None, "error": None}],
        description=f"{len(failed)}/{len(results)} files failed",
    )

    structured = [
        r.structured_metadata for r in succeeded if r.structured_metadata is not None
    ]

    _SAMPLE = 5
    sample_df = (
        polars.DataFrame(
            data=[vars(s) for s in structured[:_SAMPLE]],
            schema=StructuredMetadata.polars_schema,
        )
        if structured
        else polars.DataFrame(schema=StructuredMetadata.polars_schema)
    )
    prefect.artifacts.create_table_artifact(
        key="structured-metadata-sample",
        table=sample_df.to_dicts(),
        description=f"First {min(_SAMPLE, len(structured))} rows of structured metadata ({len(structured)} total)",
    )

    if succeeded:
        sample_result = succeeded[0]
        if sample_result.unstructured_metadata is not None:
            prefect.artifacts.create_table_artifact(
                key="unstructured-metadata-sample",
                table=[
                    {
                        "s3_uri": sample_result.s3_uri,
                        "unstructured_metadata": str(
                            sample_result.unstructured_metadata.load()
                        ),
                    }
                ],
                description=f"Unstructured metadata sample from {sample_result.s3_uri}",
            )

    return results
