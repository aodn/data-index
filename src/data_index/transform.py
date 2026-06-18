from __future__ import annotations

import logging
import typing

import prefect
import prefect.cache_policies

from data_index.protocols import (
    ExtractionResult,
    MetadataExtractor,
    UnstructuredMetadata,
    XarrayHandle,
)
from data_index.unstructured_metadata import DiskCachedUnstructuredMetadata


def _transform_single(
    xarray_handle: XarrayHandle,
    extractor: MetadataExtractor,
    unstructured_metadata_factory: typing.Callable[[str, dict], UnstructuredMetadata],
    logger: logging.Logger,
) -> ExtractionResult:
    # Attempt extraction
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

    # Attempt cleanup
    finally:
        try:
            xarray_handle.ds.close()
        except Exception as exc:
            logger.warning(
                f"Disposal of xarray handle failed for {xarray_handle.s3_uri}: {exc}"
            )


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

    Runs _transform_single sequentially. Each call immediately persists
    unstructured metadata via metadata_factory(s3_uri, data). Releases handle
    resources after all files are processed.

    Args:
        max_workers: Retained for API compatibility. No effect when running
            sequentially.

    Returns list of ExtractionResult (succeeded and failed). Callers route to sinks.
    """
    logger = prefect.get_run_logger()

    logger.info("Running extraction sequentially...")
    results = [
        _transform_single(
            xarray_handle=xarray_handle,
            extractor=extractor,
            unstructured_metadata_factory=metadata_factory,
            logger=logger,
        )
        for xarray_handle in xarray_handles
    ]
    logger.info("Sequential extraction complete!")

    logger.info("Transform complete!")
    return results
