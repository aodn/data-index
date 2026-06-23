from __future__ import annotations

import collections

import prefect
import prefect.artifacts
import prefect.cache_policies

from data_index.protocols import FileFetcher, ObjectReference


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE)
def extract(
    object_references: list[ObjectReference],
    fetcher: FileFetcher,
) -> list[ObjectReference]:
    """
    Sync a batch of S3 NetCDF files to local disk.

    Validates: required schema columns, unique object versions, total size within limit.
    Returns a list of XarrayHandle per file.
    """

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

    # Fetch
    object_references = fetcher.fetch(object_references=object_references)
    return object_references
