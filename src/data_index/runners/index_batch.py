import contextlib

import prefect
import prefect.task_runners

import data_index
import data_index.protocols
from data_index.runners.types import (
    FileFetcher,
    MetadataExtractor,
    MetadataSink,
)


@contextlib.contextmanager
def etl_phase(phase_name: str):
    """Context manager to handle standardized phase logging."""
    logger = prefect.get_run_logger()
    logger.info(f"{phase_name.capitalize()}ing batch...")

    yield

    logger.info(f"{phase_name.capitalize()}ed batch!")


@prefect.task
def sink_dead_letters(
    dead_letters: list[data_index.protocols.DeadLetter],
    dead_letter_sink: data_index.protocols.MetadataSink,
) -> None:

    if not dead_letters:
        return

    logger = prefect.get_run_logger()
    logger.error(f"Found {len(dead_letters)} dead letters!")
    logger.info(f"writing {len(dead_letters)} dead letters...")
    dead_letter_sink.write(metadata=dead_letters)
    logger.info(f"Wrote {len(dead_letters)} dead letters!")


@prefect.flow(
    task_runner=prefect.task_runners.ProcessPoolTaskRunner(
        max_workers=16,
    ),
)
def index_batch(
    compressed_object_reference_batch: str,
    fetcher: FileFetcher,
    extractor: MetadataExtractor,
    structured_sink: MetadataSink,
    unstructured_sink: MetadataSink,
    dead_letter_sink: MetadataSink,
    max_workers: int | None = None,
) -> None:
    """Full ETL pipeline for a single Batch, dispatched as a worker task."""

    total_dead_letters = 0

    # Decompress batch
    object_reference_batch = (
        data_index.protocols.ObjectReference.from_compressed_base64_table(
            base64_str=compressed_object_reference_batch,
        )
    )

    # Extract batch
    with etl_phase(phase_name="extract"):
        staged_objects, dead_letters = data_index.extract(
            object_references=object_reference_batch,
            fetcher=fetcher,
        )
    if dead_letters:
        sink_dead_letters(dead_letters=dead_letters, dead_letter_sink=dead_letter_sink)
        total_dead_letters += len(dead_letters)

    # Transform batch
    with etl_phase(phase_name="transform"):
        extracted_objects, dead_letters = data_index.transform(
            staged_objects=staged_objects,
            extractor=extractor,
            max_workers=max_workers,
        )
    if dead_letters:
        sink_dead_letters(dead_letters=dead_letters, dead_letter_sink=dead_letter_sink)
        total_dead_letters += len(dead_letters)

    # Load batch
    with etl_phase(phase_name="load"):
        dead_letters = data_index.load(
            extracted_objects=extracted_objects,
            structured_sink=structured_sink,
            unstructured_sink=unstructured_sink,
        )
    if dead_letters:
        sink_dead_letters(dead_letters=dead_letters, dead_letter_sink=dead_letter_sink)
        total_dead_letters += len(dead_letters)

    if total_dead_letters:
        raise RuntimeError(
            f"Sent {total_dead_letters} dead letters to the dead letter sink!"
        )


if __name__ == "__main__":
    index_batch.serve(
        name="index-batch",
        global_limit=12,
    )
