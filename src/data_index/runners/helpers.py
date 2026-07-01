import prefect

import data_index.protocols


def _split_object_reference_batch(
    object_reference_batch: list[data_index.protocols.ObjectReference],
    max_size_bytes: int = 512 * 1024,
) -> list[str]:
    """
    Recursively splits a list of data_index.protocols.ObjectReferences until each batch
    compressed base64 string is under the specified byte limit.
    """
    compressed_batch = data_index.protocols.ObjectReference.to_compressed_base64_table(
        object_references=object_reference_batch
    )

    # Check size
    if len(compressed_batch.encode("utf-8")) <= max_size_bytes:
        return [compressed_batch]

    # If a single item is already over the limit, we cannot split further
    if len(object_reference_batch) <= 1:
        raise ValueError("Single object reference exceeds the 512KB limit.")

    # Split the list in half
    mid = len(object_reference_batch) // 2
    left_batch = object_reference_batch[:mid]
    right_batch = object_reference_batch[mid:]

    # Recurse
    return _split_object_reference_batch(
        left_batch, max_size_bytes
    ) + _split_object_reference_batch(right_batch, max_size_bytes)


@prefect.task()
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
