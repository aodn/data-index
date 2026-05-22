from __future__ import annotations

import pathlib

import polars

from data_index.iceberg_table_config import IcebergTableConfig
from data_index.s3_metadata.extract import TableScanConfig, extract
from data_index.s3_metadata.load import load
from data_index.s3_metadata.transform import transform

_DEFAULT_PATH = pathlib.Path(".extract/s3_metadata")
_INVENTORY_PARQUET = pathlib.Path("imos-data.inventory.parquet")


class LiveS3InventorySource:
    """InventorySource that runs the s3_metadata ETL to materialise the live S3 inventory
    table to disk, then reads it back as a DataFrame with `s3_uri` and `size` columns.

    If skip_if_exists=True (default) and the target path already contains parquet files,
    the ETL is skipped and the existing data is returned directly.
    """

    def __init__(
        self,
        table_config: IcebergTableConfig,
        table_scan_config: TableScanConfig = TableScanConfig(),
        path: pathlib.Path = _DEFAULT_PATH,
        skip_if_exists: bool = True,
    ) -> None:
        self._table_config = table_config
        self._table_scan_config = table_scan_config
        self._path = path
        self._skip_if_exists = skip_if_exists

    def _has_data(self) -> bool:
        return self._path.exists() and any(self._path.rglob("*.parquet"))

    def _run_etl(self) -> None:
        inventory_lf = extract(
            table_config=self._table_config,
            table_scan_config=self._table_scan_config,
            inventory_parquet_path=_INVENTORY_PARQUET,
        )
        live_lf = transform(inventory_lf)
        load(live_lf, path=self._path)

    def inventory(self) -> polars.DataFrame:
        if not (self._skip_if_exists and self._has_data()):
            self._run_etl()

        return (
            polars.scan_parquet(self._path / "**" / "*.parquet")
            .select(
                polars.concat_str(
                    polars.lit("s3://"),
                    polars.col("bucket"),
                    polars.lit("/"),
                    polars.col("key"),
                ).alias("s3_uri"),
                polars.col("size"),
            )
            .collect()
        )
