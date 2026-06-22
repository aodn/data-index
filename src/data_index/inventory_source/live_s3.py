from __future__ import annotations

import pathlib
import typing

import polars
import pydantic

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig
from data_index.inventory_source._contract import enforce_inventory_contract
from data_index.s3_metadata.extract import extract
from data_index.s3_metadata.load import load
from data_index.s3_metadata.transform import transform

_DEFAULT_PATH = pathlib.Path(".extract/s3_metadata")
_INVENTORY_PARQUET = pathlib.Path("imos-data.inventory.parquet")


class LiveS3InventorySource(pydantic.BaseModel):
    """InventorySource that runs the s3_metadata ETL to materialise the live S3 inventory
    table to disk, then reads it back as a DataFrame with required identity columns.

    If skip_if_exists=True (default) and the target path already contains parquet files,
    the ETL is skipped and the existing data is returned directly.
    """

    type: typing.Literal["live_s3"] = pydantic.Field(default="live_s3")

    table_config: IcebergTableConfig
    table_scan_config: IcebergTableScanConfig
    path: pathlib.Path = pydantic.Field(
        default_factory=lambda: pathlib.Path(_DEFAULT_PATH)
    )
    skip_if_exists: bool = pydantic.Field(default=True)

    def _has_data(self) -> bool:
        return self.path.exists() and any(self.path.rglob("*.parquet"))

    def _run_etl(self) -> None:
        inventory_lf = extract(
            table_config=self.table_config,
            table_scan_config=self.table_scan_config,
            inventory_parquet_path=_INVENTORY_PARQUET,
        )
        live_lf = transform(inventory_lf)
        load(live_lf, path=self.path)

    def inventory(self) -> polars.DataFrame:
        if not (self.skip_if_exists and self._has_data()):
            self._run_etl()

        return enforce_inventory_contract(
            polars.scan_parquet(self.path / "**" / "*.parquet")
            .select(
                polars.col("bucket"),
                polars.col("key"),
                polars.col("version_id"),
                polars.col("size"),
            )
            .collect()
        )
