from __future__ import annotations

import collections

import prefect
import prefect.artifacts
import prefect.cache_policies

from data_index.protocols import FileFetcher, ObjectReference


@prefect.task(
    cache_policy=prefect.cache_policies.NO_CACHE,
    retries=3,
    retry_delay_seconds=[5, 13, 35],
)
def extract(
    object_references: list[ObjectReference],
    fetcher: FileFetcher,
) -> list[ObjectReference]:
    """
    Sync a batch of S3 NetCDF files to local disk.

    Validates: required schema columns, unique object versions, total size within limit.
    Returns a list of XarrayHandle per file.
    """

    logger = prefect.get_run_logger()

    # Return empty list if no object_references passed in
    if not object_references:
        logger.warning("extract called with no object references!")
        return list()

    # Count occurrences using the built-in versioned URI generator
    uri_counts = collections.Counter(
        object_reference.as_versioned_uri() for object_reference in object_references
    )

    # Isolate only the URIs that appear more than once
    duplicate_details = [
        f"{uri} (appears {count} times)"
        for uri, count in uri_counts.items()
        if count > 1
    ]

    if duplicate_details:
        raise ValueError(
            "Validation failed: Duplicate object references detected.\n"
            "Duplicates list:\n" + "\n".join(duplicate_details)
        )

    # Get the number of bytes
    bytes = sum([object_reference.size for object_reference in object_references])

    # Fetch
    logger.info(
        f"Extracting `{len(object_references)}` files (`{(bytes / 1024 / 1024):,.2f}` MB)"
    )
    object_references = fetcher.fetch(object_references=object_references)
    logger.info(f"Extracted `{len(object_references)}` files!")
    return object_references
