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
    succeeded = [r for r in extraction_results if r.status == "succeeded"]

    structured = [
        result.structured_metadata
        for result in succeeded
        if result.structured_metadata is not None
    ]
    structured_sink.write(structured)
    logger.info(f"Wrote {len(structured)} structured metadata rows")

    # Re-Hydrate unstructured metadata
    # TODO: May need to occur in batches to not overflow
    unstructured = {
        r.s3_uri: r.unstructured_metadata.load()
        for r in succeeded
        if r.unstructured_metadata is not None
    }
    if unstructured:
        unstructured_sink.write(unstructured)
        logger.info(f"Wrote {len(unstructured)} unstructured metadata entries")
