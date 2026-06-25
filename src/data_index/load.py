from __future__ import annotations

import prefect
import prefect.cache_policies

import data_index.protocols


@prefect.task(
    cache_policy=prefect.cache_policies.NO_CACHE,
    retries=3,
    retry_delay_seconds=[5, 13, 35],
)
def load(
    extracted_objects: list[data_index.protocols.ExtractedObject],
    structured_sink: data_index.protocols.Sink,
    unstructured_sink: data_index.protocols.Sink,
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
        return list()

    # TODO: DLQ
    # Try catch logic here; if the load fails then add all object_references to DLQ
    logger.info("Sinking structued metadata rows...")
    dead_letters = structured_sink.write(extracted_objects)
    logger.info(f"Sunk {len(extracted_objects)} structured metadata rows!")
    # End catch;

    # TODO: DLQ
    # Try catch logic here; if the load fails then add all object_references to DLQ
    logger.info("Sinking unstructued metadata rows...")
    dead_letters = unstructured_sink.write(extracted_objects)  # noqa
    logger.info(f"Sunk {len(extracted_objects)} unstructured metadata rows!")
    # End Catch;
