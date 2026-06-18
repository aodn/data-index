from __future__ import annotations

import typing

import polars
import pydantic

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig
from data_index.inventory_source.live_s3 import LiveS3InventorySource


class S3TableInventorySource(LiveS3InventorySource):
    """InventorySource that runs on a pre-live-evaluated S3 Metadata Table."""

    type: typing.Literal["s3_table"] = pydantic.Field(default="s3_table")

    table_config: IcebergTableConfig
    table_scan_config: IcebergTableScanConfig = pydantic.Field(
        default_factory=IcebergTableScanConfig
    )

    @staticmethod
    def _s3_uri_column() -> polars.Expr:
        return polars.concat_str(
            polars.lit("s3://"),
            polars.col("bucket"),
            polars.lit("/"),
            polars.col("key"),
        ).alias("s3_uri")

    @staticmethod
    def _empty_inventory() -> polars.DataFrame:
        return polars.DataFrame(schema={"s3_uri": polars.String, "size": polars.Int64})

    def _scan(self, selected_fields: tuple[str, ...]) -> polars.DataFrame:
        scan_kwargs = self.table_scan_config.model_dump(exclude_none=True)
        scan_kwargs["selected_fields"] = selected_fields

        table = self.table_config.load()
        return table.scan(**scan_kwargs).to_polars()

    def inventory(self) -> polars.DataFrame:
        df = self._scan(selected_fields=("bucket", "key", "size"))
        if df.is_empty():
            return self._empty_inventory()

        return df.select(
            self._s3_uri_column(),
            polars.col("size"),
        )


class S3TableFacilitySubsetInventorySource(S3TableInventorySource):
    """S3 table inventory source with per-facility random sampling."""

    type: typing.Literal["s3_table_facility_subset"] = pydantic.Field(
        default="s3_table_facility_subset"
    )
    subset_per_facility: int = pydantic.Field(default=10_000, ge=1)

    def inventory(self) -> polars.DataFrame:
        df = self._scan(selected_fields=("bucket", "key", "size", "facility"))
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

        return polars.concat(sampled_slices).select(
            self._s3_uri_column(),
            polars.col("size"),
        )
