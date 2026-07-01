import polars
import pydantic

import data_index.inventory_source.iceberg_table


class DeltaIcebergTableInventorySource(pydantic.BaseModel):
    source: data_index.inventory_source.iceberg_table.IcebergTableInventorySource
    sink: data_index.inventory_source.iceberg_table.IcebergTableInventorySource
    left_on: tuple[str] = pydantic.Field(default=("bucket", "key", "version_id"))
    right_on: tuple[str] = pydantic.Field(default=("bucket", "key", "version_id"))

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
        source=data_index.inventory_source.iceberg_table.IcebergTableInventorySource(
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
        sink=data_index.inventory_source.iceberg_table.IcebergTableInventorySource(
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
    inventory = inventory_source.inventory()
    rich.print(inventory)
