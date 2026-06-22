from __future__ import annotations

import typing

import polars
import pydantic

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig
from data_index.inventory_source._contract import enforce_inventory_contract
from data_index.inventory_source.live_s3 import LiveS3InventorySource


class S3TableInventorySource(LiveS3InventorySource):
    """InventorySource that runs on a pre-live-evaluated S3 Metadata Table."""

    type: typing.Literal["s3_table"] = pydantic.Field(default="s3_table")

    table_config: IcebergTableConfig
    table_scan_config: IcebergTableScanConfig = pydantic.Field(
        default_factory=IcebergTableScanConfig
    )

    @staticmethod
    def _empty_inventory() -> polars.DataFrame:
        return polars.DataFrame(
            schema={
                "bucket": polars.String,
                "key": polars.String,
                "version_id": polars.String,
                "size": polars.Int64,
            }
        )

    def _scan(self, selected_fields: tuple[str, ...]) -> polars.DataFrame:
        scan_kwargs = self.table_scan_config.model_dump(exclude_none=True)
        scan_kwargs["selected_fields"] = selected_fields

        table = self.table_config.load()
        return table.scan(**scan_kwargs).to_polars()

    def inventory(self) -> polars.DataFrame:
        df = self._scan(selected_fields=("bucket", "key", "version_id", "size"))
        if df.is_empty():
            return self._empty_inventory()

        return enforce_inventory_contract(
            df.select(
                polars.col("bucket"),
                polars.col("key"),
                polars.col("version_id"),
                polars.col("size"),
            )
        )


class S3TableFacilitySubsetInventorySource(S3TableInventorySource):
    """S3 table inventory source with per-facility random sampling."""

    type: typing.Literal["s3_table_facility_subset"] = pydantic.Field(
        default="s3_table_facility_subset"
    )
    subset_per_facility: int = pydantic.Field(default=10_000, ge=1)

    def inventory(self) -> polars.DataFrame:
        df = self._scan(
            selected_fields=("bucket", "key", "version_id", "size", "facility")
        )
        if df.is_empty():
            return self._empty_inventory()

        sampled_slices: list[polars.DataFrame] = []
        for facility_slice in df.drop_nulls("facility").partition_by("facility"):
            sample_n = min(self.subset_per_facility, facility_slice.height)
            sampled_slices.append(
                facility_slice.sample(n=sample_n, with_replacement=False, shuffle=True)
            )

        if not sampled_slices:
            return self._empty_inventory()

        return enforce_inventory_contract(
            polars.concat(sampled_slices).select(
                polars.col("bucket"),
                polars.col("key"),
                polars.col("version_id"),
                polars.col("size"),
            )
        )
