from __future__ import annotations

import prefect

from data_index.protocols import ExtractionResult, StructuredSink, UnstructuredSink


@prefect.task
def load(
    extraction_results: list[ExtractionResult],
    structured_sink: StructuredSink,
    unstructured_sink: UnstructuredSink | None = None,
) -> None:
    """Persist structured and unstructured metadata via injected sinks.

    Structured metadata is always written. Unstructured metadata is written only
    if unstructured_sink is provided — each entry's data is resolved via .load()
    before being passed to the sink.
    """
    logger = prefect.get_run_logger()
    succeeded = [r for r in extraction_results if r.status == "succeeded"]

    structured = [r.structured_metadata for r in succeeded if r.structured_metadata is not None]
    structured_sink.write(structured)
    logger.info(f"Wrote {len(structured)} structured metadata rows")

    if unstructured_sink is not None:
        unstructured = {
            r.s3_uri: r.unstructured_metadata.load()
            for r in succeeded
            if r.unstructured_metadata is not None
        }
        if unstructured:
            unstructured_sink.write(unstructured)
            logger.info(f"Wrote {len(unstructured)} unstructured metadata entries")
