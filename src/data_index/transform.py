from __future__ import annotations

import pathlib
import typing

import polars
import prefect
import prefect.artifacts
import prefect.futures

from data_index.protocols import (
    ExtractionResult,
    XarrayHandle,
    MetadataExtractor,
    StructuredMetadata,
    UnstructuredMetadata,
)
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


@prefect.task(
    task_run_name="transform-single-{xarray_handle.s3_uri}",
)
def _transform_single(
    xarray_handle: XarrayHandle,
    extractor: MetadataExtractor,
    unstructured_metadata_factory: typing.Callable[[str, dict], UnstructuredMetadata],
) -> ExtractionResult:
    ds = xarray_handle.ds
    try:
        raw = extractor.extract(ds=ds, s3_uri=xarray_handle.s3_uri)
        if raw.status == "failed":
            return ExtractionResult(
                s3_uri=xarray_handle.s3_uri,
                structured_metadata=None,
                unstructured_metadata=None,
                status="failed",
                error=raw.error,
            )
        return ExtractionResult(
            s3_uri=xarray_handle.s3_uri,
            structured_metadata=raw.structured_metadata,
            unstructured_metadata=unstructured_metadata_factory(xarray_handle.s3_uri, raw.unstructured_metadata),
            status="succeeded",
        )
    except Exception as exc:
        return ExtractionResult(
            s3_uri=xarray_handle.s3_uri,
            structured_metadata=None,
            unstructured_metadata=None,
            status="failed",
            error=str(exc),
        )
    finally:
        ds.close()


@prefect.task
def transform(
    xarray_handles: list[XarrayHandle],
    extractor: MetadataExtractor,
    metadata_factory: typing.Callable[[str, dict], UnstructuredMetadata] = DiskCachedUnstructuredMetadata,
) -> list[ExtractionResult]:
    """
    Transform a manifest of local NetCDF files into structured and unstructured metadata.

    Submits one _transform_single task per file (parallel). Each task immediately
    persists unstructured metadata via metadata_factory(s3_uri, data). Deletes local
    files after all tasks complete.

    Returns list of ExtractionResult (succeeded and failed). Callers route to sinks.
    """
    futures = [
        _transform_single.submit(
            xarray_handle=xarray_handle,
            extractor=extractor,
            unstructured_metadata_factory=metadata_factory,
        )
        for xarray_handle in xarray_handles
    ]

    results = []
    for future in prefect.futures.as_completed(futures=futures):
        results.append(future.result())

    # Release handle resources (e.g. delete local files for DiskXarrayHandle)
    for xarray_handle in xarray_handles:
        xarray_handle.cleanup()

    failed = [r for r in results if r.status == "failed"]
    succeeded = [r for r in results if r.status == "succeeded"]

    if failed:
        logger = prefect.get_run_logger()
        for r in failed:
            logger.warning(f"transform failed for {r.s3_uri}: {r.error}")

    prefect.artifacts.create_table_artifact(
        key="transform-succeeded",
        table=[{"s3_uri": r.s3_uri} for r in succeeded],
        description=f"{len(succeeded)}/{len(results)} files succeeded",
    )
    prefect.artifacts.create_table_artifact(
        key="transform-failed",
        table=[{"s3_uri": r.s3_uri, "error": r.error} for r in failed] or [{"s3_uri": None, "error": None}],
        description=f"{len(failed)}/{len(results)} files failed",
    )

    structured = [r.structured_metadata for r in succeeded if r.structured_metadata is not None]

    _SAMPLE = 5
    sample_df = polars.DataFrame(
        [vars(s) for s in structured[:_SAMPLE]],
        schema=StructuredMetadata.polars_schema,
    ) if structured else polars.DataFrame(schema=StructuredMetadata.polars_schema)
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
                table=[{"s3_uri": sample_result.s3_uri, "unstructured_metadata": str(sample_result.unstructured_metadata.load())}],
                description=f"Unstructured metadata sample from {sample_result.s3_uri}",
            )

    return results
