from __future__ import annotations

import pathlib
import typing

import polars
import pydantic

from data_index.inventory_source import LiveS3InventorySource
from data_index.inventory_source._contract import enforce_inventory_contract

_DEFAULT_PATH = pathlib.Path(".extract/s3_metadata")
_INVENTORY_PARQUET = pathlib.Path("imos-data.inventory.parquet")


class LiveS3InventorySourceFacilitySubset(LiveS3InventorySource):
    """InventorySource that runs the s3_metadata ETL to materialise the live S3 inventory
    table to disk, then returns required identity columns plus size.

    If skip_if_exists=True (default) and the target path already contains parquet files,
    the ETL is skipped and the existing data is returned directly.
    """

    type: typing.Literal["live_s3_facility_subset"] = pydantic.Field(
        default="live_s3_facility_subset"
    )

    subset_per_facility: int = pydantic.Field(default=10_000, ge=1)

    def inventory(self) -> polars.DataFrame:
        """
        Return a subset from each facility
        """
        if not (self.skip_if_exists and self._has_data()):
            self._run_etl()

        inventory_scan = polars.scan_parquet(self.path / "**" / "*.parquet")

        # Find facilities, defined as second part of path
        facilities = (
            inventory_scan.select(
                polars.col("key")
                .str.extract(pattern=r"^IMOS/([^/]+)/", group_index=1)
                .alias("facility"),
            )
            .drop_nulls()
            .unique()
            .collect()
            .get_column("facility")
            .to_list()
        )

        # Takes samples of size `subset_per_facility` per facility
        sampled_slices: list[polars.DataFrame] = []
        for facility in facilities:
            facility_rows = (
                polars.scan_parquet(self.path / "**" / "*.parquet")
                .filter(polars.col("key").str.starts_with(f"IMOS/{facility}/"))
                .select("bucket", "key", "version_id", "size")
                .collect()
            )
            if facility_rows.is_empty():
                continue

            sample_n = min(self.subset_per_facility, facility_rows.height)
            sampled_slices.append(
                facility_rows.sample(n=sample_n, with_replacement=False, shuffle=True)
            )

        # No samples case
        if not sampled_slices:
            return polars.DataFrame(
                schema={
                    "bucket": polars.String,
                    "key": polars.String,
                    "version_id": polars.String,
                    "size": polars.Int64,
                }
            )

        return enforce_inventory_contract(
            polars.concat(sampled_slices).select(
                polars.col("bucket"),
                polars.col("key"),
                polars.col("version_id"),
                polars.col("size"),
            )
        )
