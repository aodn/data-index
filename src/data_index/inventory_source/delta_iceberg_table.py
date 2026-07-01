from __future__ import annotations

import datetime
import typing

import polars
import pydantic

from data_index.inventory_source import iceberg_table


class LookbackConfig(pydantic.BaseModel):
    days: int = pydantic.Field(default=0)
    hours: int = pydantic.Field(default=0)
    column_name: str = pydantic.Field(default="last_modified_date")

    @property
    def lookback_timestamp(self):
        return (
            datetime.datetime.now(tz=datetime.timezone.utc)
            - datetime.timedelta(days=self.days, hours=self.hours)
        ).isoformat()

    @property
    def lookback_filter(self):
        return f"{self.column_name} >= '{self.lookback_timestamp}'"


class DeltaIcebergTableInventorySource(pydantic.BaseModel):
    type: typing.Literal["delta_iceberg_table"] = pydantic.Field(
        default="delta_iceberg_table"
    )

    source: iceberg_table.IcebergTableInventorySource
    sink: iceberg_table.IcebergTableInventorySource
    lookback_config: LookbackConfig | None = pydantic.Field(
        default=None,
        description="The number of hours to look back from the current time in the given column",
    )
    left_on: list[str] = pydantic.Field(default=["bucket", "key", "version_id"])
    right_on: list[str] = pydantic.Field(default=["bucket", "key", "version_id"])

    @pydantic.model_validator(mode="after")
    def apply_lookback_config(self) -> typing.Self:
        """Applies the lookback configuration filters to the source table scan config."""
        if self.lookback_config:
            if self.source.table_scan_config.row_filter:
                self.source.table_scan_config.row_filter = f"({self.source.table_scan_config.row_filter}) AND {self.lookback_config.lookback_filter}"
            else:
                self.source.table_scan_config.row_filter = (
                    self.lookback_config.lookback_filter
                )

        return self

    def inventory(
        self,
    ) -> polars.DataFrame:
        source_df = self.source.inventory()
        sink_df = self.sink.inventory()

        delta_df = source_df.join(
            other=sink_df,
            left_on=self.left_on,
            right_on=self.right_on,
            how="anti",
        )

        return delta_df


if __name__ == "__main__":
    import datetime

    import rich

    import data_index.iceberg_config

    # Instead of a lambda, a named function makes your code much easier to read and debug:
    def get_lookback_timestamp(time_delta: datetime.timedelta) -> str:
        return datetime.datetime.combine(
            date=datetime.date.today() - time_delta,
            time=datetime.time.min,
        ).isoformat()

    inventory_source = DeltaIcebergTableInventorySource(
        source=iceberg_table.IcebergTableInventorySource(
            table_config=data_index.iceberg_config.IcebergTableConfig(
                catalog_config=data_index.iceberg_config.S3TablesCatalogConfig(
                    region="ap-southeast-2",
                    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
                ),
                namespace="inventory",
                table_name="live",
            ),
            table_scan_config=data_index.iceberg_config.IcebergTableScanConfig(
                selected_fields=("bucket", "key", "version_id", "size"),
                row_filter=f"last_modified_date >= '{get_lookback_timestamp(time_delta=datetime.timedelta(days=10))}'",
            ),
        ),
        sink=iceberg_table.IcebergTableInventorySource(
            table_config=data_index.iceberg_config.IcebergTableConfig(
                catalog_config=data_index.iceberg_config.S3TablesCatalogConfig(
                    region="ap-southeast-2",
                    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
                ),
                namespace="data_index",
                table_name="structured_metadata_v5",
            ),
            table_scan_config=data_index.iceberg_config.IcebergTableScanConfig(
                selected_fields=("bucket", "key", "version_id"),
            ),
        ),
    )

    rich.print(inventory_source)
    rich.print(inventory_source.model_dump_json(indent=4))
    # inventory = inventory_source.inventory()
    # rich.print(inventory)
