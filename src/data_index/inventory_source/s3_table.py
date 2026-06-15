from __future__ import annotations

import typing

import polars
import pydantic

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig
from data_index.inventory_source import LiveS3InventorySource


class S3TableInventorySource(LiveS3InventorySource):
    """InventorySource that runs on a pre-live-evaluated S3 Metadata Table."""

    type: typing.Literal["s3_table"] = pydantic.Field(default="s3_table")

    subset_per_facility: int = pydantic.Field(default=10_000, ge=1)
    table_config: IcebergTableConfig
    table_scan_config: IcebergTableScanConfig = pydantic.Field(
        default_factory=lambda: IcebergTableScanConfig(
            selected_fields=("bucket", "key", "size", "facility"),
        )
    )
    subset_per_facility: int | None = pydantic.Field(default=None)

    @staticmethod
    def _s3_uri_column() -> polars.Expr:
        return polars.concat_str(
            polars.lit("s3://"),
            polars.col("bucket"),
            polars.lit("/"),
            polars.col("key"),
        ).alias("s3_uri")

    def _full_inventory(self) -> polars.DataFrame:
        table = self.table_config.load()
        df = table.scan(selected_fields=("bucket", "key", "size")).to_polars()
        return df.select(
            self._s3_uri_column(),
            polars.col("size"),
        )

    def _subset_per_facility(self) -> polars.DataFrame:
        table = self.table_config.load()

        facilities_df = table.scan(selected_fields=("facility",)).to_polars()

        # Find facilities, defined as second part of path
        facilities = facilities_df["facility"].drop_nulls().unique()

        # Takes samples of size `subset_per_facility` per facility
        sampled_slices: list[polars.DataFrame] = []
        for facility in facilities:
            df = table.scan(
                selected_fields=("bucket", "key", "size"),
                row_filter=f"facility = '{facility}'",
            ).to_polars()
            if df.is_empty():
                continue

            sample_n = min(self.subset_per_facility, df.height)
            sampled_slices.append(
                df.sample(n=sample_n, with_replacement=False, shuffle=True)
            )

        # No samples case
        if not sampled_slices:
            return polars.DataFrame(
                schema={"s3_uri": polars.String, "size": polars.Int64}
            )

        # Concat and adjust to s3_uri
        return polars.concat(sampled_slices).select(
            self._s3_uri_column(),
            polars.col("size"),
        )

    def inventory(self) -> polars.DataFrame:
        """
        Return a subset from each facility
        """

        if self.subset_per_facility:
            return self._subset_per_facility()
        else:
            return self._full_inventory()
