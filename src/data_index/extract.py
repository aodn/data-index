from __future__ import annotations

import pathlib

import polars
import prefect
import prefect.artifacts

from data_index.protocols import FileFetcher, XarrayHandle


@prefect.task
def extract(
    batch_df: polars.DataFrame,
    fetcher: FileFetcher,
    batch_schema: polars.Schema = polars.Schema(
        schema={
            "s3_uri": polars.String,
            "size": polars.Int64(),
        }
    ),
    batch_size_limit: int = 50 * 2**30,  # 50GB
) -> list[XarrayHandle]:
    """
    Sync a batch of S3 NetCDF files to local disk.

    Validates: schema, unique s3_uris, total size within limit.
    Returns a list of XarrayHandle with s3_uri and absolute_path per file.
    """

    logger = prefect.get_run_logger()

    if batch_df.schema != batch_schema:
        raise ValueError(f"Batch schema mismatch: expected {batch_schema}, got {batch_df.schema}")

    if batch_df["s3_uri"].null_count() > 0:
        raise ValueError("Null s3_uris in batch")

    if batch_df["s3_uri"].n_unique() != len(batch_df):
        raise ValueError("Duplicate s3_uris in batch")

    total_size = batch_df["size"].sum()
    if total_size > batch_size_limit:
        raise ValueError(f"Batch size {total_size} bytes exceeds limit {batch_size_limit}")
    else:
        logger.info(f"Processing batch of `{len(batch_df)}` files => `{round(batch_df["size"].sum() / 2 ** 30, 2)}`GB")
    # Fetch
    manifest = fetcher.fetch(batch_df["s3_uri"].to_list())

    # Report manifest
    prefect.artifacts.create_table_artifact(
        key="extract-manifest",
        table=[
            manifest_entry.model_dump(mode="json")
            for manifest_entry in manifest
        ],
        description="The extraction manifest"
    )

    return manifest