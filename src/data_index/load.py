import typing

import prefect
import prefect.cache_policies

import data_index.protocols


@prefect.task(
    task_run_name="sink-{metadata_type}",
    cache_policy=prefect.cache_policies.NO_CACHE,
    retries=4,
    retry_delay_seconds=[30, 60, 120, 300],
)
def sink(
    extracted_objects: list[data_index.protocols.ExtractedObject],
    metadata_type: typing.Literal["structured_metadata", "unstructured_metadata"],
    sink: data_index.protocols.MetadataSink,
) -> None:

    logger = prefect.get_run_logger()

    # Split the metadata_type
    metadata = [
        getattr(extracted_object.extraction_result, metadata_type)
        for extracted_object in extracted_objects
    ]

    # Attempt to sink
    logger.info(f"Sinking {len(extracted_objects)} {metadata_type} rows...")
    sink.write(metadata=metadata)
    logger.info(f"Sunk {len(extracted_objects)} {metadata_type} metadata rows!")


@prefect.task(
    cache_policy=prefect.cache_policies.NO_CACHE,
    retries=2,
    retry_delay_seconds=[10],
)
def load(
    extracted_objects: list[data_index.protocols.ExtractedObject],
    structured_sink: data_index.protocols.MetadataSink,
    unstructured_sink: data_index.protocols.MetadataSink,
) -> list[data_index.protocols.DeadLetter]:
    """Persist structured and unstructured metadata via injected sinks.

    Structured metadata is always written. Unstructured metadata is written only
    if unstructured_sink is provided — each entry's data is resolved via .load()
    before being passed to the sink.
    """
    logger = prefect.get_run_logger()

    # Return empty list if no object_references passed in
    if not extracted_objects:
        logger.warning("load called with no extraction results!")
        return []

    # Sink structured metadata
    state = sink(
        extracted_objects=extracted_objects,
        metadata_type="structured_metadata",
        sink=structured_sink,
        return_state=True,
    )

    if state.is_failed():
        logger.error(f"Structured sink completely failed: {state.message}")
        return [
            data_index.protocols.DeadLetter.from_object_reference(
                object_reference=extracted_object.object_reference,
                error=str(state.message),
            )
            for extracted_object in extracted_objects
        ]

    # Sink structured metadata
    state = sink(
        extracted_objects=extracted_objects,
        metadata_type="unstructured_metadata",
        sink=unstructured_sink,
        return_state=True,
    )
    if state.is_failed():
        logger.error(f"Unstructured sink completely failed: {state.message}")
        return [
            data_index.protocols.DeadLetter.from_object_reference(
                object_reference=extracted_object.object_reference,
                error=str(state.message),
            )
            for extracted_object in extracted_objects
        ]

    # Return empty list if no dead letters
    return []
