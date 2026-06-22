from __future__ import annotations

import polars
import prefect
import prefect.artifacts
import prefect.cache_policies

from data_index.protocols import BatchEntry, FileFetcher, ObjectReference, XarrayHandle


@prefect.task(cache_policy=prefect.cache_policies.NO_CACHE)
def extract(
    batch_df: polars.DataFrame,
    fetcher: FileFetcher,
    batch_schema: polars.Schema = polars.Schema(
        schema={
            "bucket": polars.String,
            "key": polars.String,
            "version_id": polars.String,
            "size": polars.Int64(),
        }
    ),
    batch_size_limit: int = 50 * 2**30,  # 50GB
) -> list[XarrayHandle]:
    """
    Sync a batch of S3 NetCDF files to local disk.

    Validates: required schema columns, unique object versions, total size within limit.
    Returns a list of XarrayHandle per file.
    """

    logger = prefect.get_run_logger()

    # Check required schema columns, allowing extras
    expected_schema = dict(batch_schema)
    missing_columns = [
        column for column in expected_schema if column not in batch_df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Batch schema missing required columns: {missing_columns}\n"
            f"Expected at least: {expected_schema}\n"
            f"Got:      {dict(batch_df.schema)}"
        )

    wrong_types = {
        column: (expected_schema[column], batch_df.schema[column])
        for column in expected_schema
        if batch_df.schema[column] != expected_schema[column]
    }
    if wrong_types:
        raise ValueError(
            f"Batch schema type mismatch for required columns:\n"
            f"Expected/Got: {wrong_types}"
        )

    for column in ("bucket", "key", "version_id"):
        if batch_df[column].null_count() > 0:
            raise ValueError(f"Null `{column}` values in batch")
        empty_identity_values = batch_df.filter(
            polars.col(column).str.strip_chars() == ""
        )
        if len(empty_identity_values) > 0:
            raise ValueError(f"Empty `{column}` values in batch")

    if batch_df["size"].null_count() > 0:
        raise ValueError("Null `size` values in batch")

    identity_columns = ["bucket", "key", "version_id"]
    if batch_df.n_unique(subset=identity_columns) != len(batch_df):
        raise ValueError("Duplicate (`bucket`, `key`, `version_id`) values in batch")

    total_size = batch_df["size"].sum()
    if total_size > batch_size_limit:
        raise ValueError(
            f"Batch size {total_size} bytes exceeds limit {batch_size_limit}"
        )
    else:
        logger.info(
            f"Processing batch of `{len(batch_df)}` files => `{round(batch_df['size'].sum() / 2**30, 2)}`GB"
        )
    # Fetch
    entries = [
        BatchEntry(
            object_ref=ObjectReference(
                bucket=row["bucket"],
                key=row["key"],
                version_id=row["version_id"],
            ),
            size_bytes=row["size"],
        )
        for row in batch_df.iter_rows(named=True)
    ]
    manifest = fetcher.fetch(entries)

    # Report manifest
    prefect.artifacts.create_table_artifact(
        key="extract-manifest",
        table=[manifest_entry.model_dump(mode="json") for manifest_entry in manifest],
        description="The extraction manifest",
    )

    return manifest
