from __future__ import annotations

import logging

import prefect
import prefect.cache_policies

import data_index.protocols


def _transform_staged_object(
    staged_object: data_index.protocols.StagedObject,
    extractor: data_index.protocols.MetadataExtractor,
    logger: logging.Logger,
) -> data_index.protocols.ExtractedObject | data_index.protocols.DeadLetter:

    # Attempt to extract the metadata from the object
    try:
        return extractor.extract(staged_object=staged_object)

    # Report dead letter if this fails
    except Exception as e:
        return data_index.protocols.DeadLetter.from_object_reference(
            object_reference=staged_object.object_reference, error=str(e)
        )

    # Clean up the object
    finally:
        try:
            staged_object.xarray_handle.cleanup()
        except Exception as e:
            logger.warning(
                f"Disposal of xarray handle failed for {staged_object.object_reference.as_versioned_uri()}: {e}"
            )


def _transform_staged_objects(
    staged_objects: list[data_index.protocols.StagedObject],
    extractor: data_index.protocols.MetadataExtractor,
    logger: logging.Logger,
) -> tuple[
    list[data_index.protocols.ExtractedObject, list[data_index.protocols.DeadLetter]]
]:
    """
    Populate all ObjectReferences with disk xarray handles.

    Causes download of all passed in object_references to `self.extract_path`
    """

    extracted_objects = [
        _transform_staged_object(
            staged_object=staged_object,
            extractor=extractor,
            logger=logger,
        )
        for staged_object in staged_objects
    ]

    return (
        [
            extracted_object
            for extracted_object in extracted_objects
            if isinstance(extracted_object, data_index.protocols.ExtractedObject)
        ],
        [
            extracted_object
            for extracted_object in extracted_objects
            if isinstance(extracted_object, data_index.protocols.DeadLetter)
        ],
    )


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE)
def transform(
    staged_objects: list[data_index.protocols.StagedObject],
    extractor: data_index.protocols.MetadataExtractor,
) -> tuple[
    list[data_index.protocols.ExtractedObject], list[data_index.protocols.DeadLetter]
]:
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

    # Return empty list if no object_references passed in
    if not staged_objects:
        logger.warning("transform called with no staged objects!")
        return list()

    logger.info("Running extraction sequentially...")

    extracted_objects, dead_letters = _transform_staged_objects(
        staged_objects=staged_objects,
        extractor=extractor,
        logger=logger,
    )
    logger.info("Sequential extraction complete!")

    logger.info("Transform complete!")
    return (
        extracted_objects,
        dead_letters,
    )
