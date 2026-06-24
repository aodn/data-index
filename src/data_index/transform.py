from __future__ import annotations

import logging

import prefect
import prefect.cache_policies

from data_index.protocols import (
    ExtractionResult,
    MetadataExtractor,
    ObjectReference,
)


def _transform_single(
    object_reference: ObjectReference,
    extractor: MetadataExtractor,
    logger: logging.Logger,
) -> ExtractionResult:

    # Attempt extraction
    try:
        extraction_result = extractor.extract(object_reference=object_reference)
        if extraction_result.status == "failed":
            logger.warning(
                f"extraction failed for {object_reference.as_versioned_uri()}: {extraction_result.error}"
            )
            return extraction_result
        logger.debug(f"extraction succeeded for {object_reference.as_versioned_uri()}")
        return extraction_result

    # Failed extraction
    except Exception as e:
        logger.warning(
            f"extraction failed for {object_reference.as_versioned_uri()}: {e}"
        )
        return ExtractionResult(
            structured_metadata=None,
            unstructured_metadata=None,
            status="error",
            error=str(e),
        )

    # Attempt cleanup
    finally:
        try:
            object_reference.xarray_handle.cleanup()
        except Exception as e:
            logger.warning(
                f"Disposal of xarray handle failed for {object_reference.as_versioned_uri()}: {e}"
            )


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE)
def transform(
    object_references: list[ObjectReference],
    extractor: MetadataExtractor,
) -> list[ExtractionResult]:
    """
    Transform a list of XarrayHandle objects into structured and unstructured metadata.

    Runs _transform_single sequentially. Each call immediately persists
    unstructured metadata via metadata_factory(object_ref, data). Releases handle
    resources after all files are processed.

    Args:
        max_workers: Retained for API compatibility. No effect when running
            sequentially.

    Returns list of ExtractionResult (succeeded and failed). Callers route to sinks.
    """
    logger = prefect.get_run_logger()

    logger.info("Running extraction sequentially...")
    extraction_results = [
        _transform_single(
            object_reference=object_reference,
            extractor=extractor,
            logger=logger,
        )
        for object_reference in object_references
    ]
    logger.info("Sequential extraction complete!")

    logger.info("Transform complete!")
    return extraction_results
