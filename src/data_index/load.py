from __future__ import annotations

import prefect
import prefect.cache_policies

from data_index.protocols import ExtractionResult, StructuredSink, UnstructuredSink


@prefect.task(
    cache_policy=prefect.cache_policies.NO_CACHE,
)
def load(
    extraction_results: list[ExtractionResult],
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink,
) -> None:
    """Persist structured and unstructured metadata via injected sinks.

    Structured metadata is always written. Unstructured metadata is written only
    if unstructured_sink is provided — each entry's data is resolved via .load()
    before being passed to the sink.
    """
    logger = prefect.get_run_logger()
    succeeded = [
        extraction_result
        for extraction_result in extraction_results
        if extraction_result.status == "succeeded"
    ]

    structured_metadata = [
        extraction_result.structured_metadata
        for extraction_result in succeeded
        if extraction_result.structured_metadata is not None
    ]
    logger.info("Sinking structued metadata rows...")
    structured_sink.write(structured_metadata)
    logger.info(f"Sunk {len(structured_metadata)} structured metadata rows!")

    # Re-Hydrate unstructured metadata
    unstructured = [
        extraction_result.unstructured_metadata
        for extraction_result in succeeded
        if extraction_result.unstructured_metadata is not None
    ]
    if unstructured:
        logger.info("Sinking unstructued metadata rows...")
        unstructured_sink.write(unstructured)
        logger.info(f"Sunk {len(unstructured)} unstructured metadata rows!")
