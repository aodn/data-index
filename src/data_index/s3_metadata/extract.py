import pathlib

import polars
import prefect
import pyarrow
import pyarrow.parquet
import pyiceberg.expressions
import pyiceberg.table
from pyiceberg.expressions import AlwaysTrue

from data_index.iceberg_config import S3TablesCatalogConfig
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig

from ._schema import INVENTORY_TABLE_SCHEMA

# Alias for backward compatibility
TableScanConfig = IcebergTableScanConfig

_SELECTED_SCHEMA_NAMES = {field.name for field in INVENTORY_TABLE_SCHEMA}


@prefect.task
def sink_table(
    table: pyiceberg.table.Table,
    table_scan_config: TableScanConfig,
    inventory_parquet_path: pathlib.Path,
):
    logger = prefect.get_run_logger()

    inventory_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Table scan config:\n{table_scan_config.model_dump_json(indent=4)}")
    scan = table.scan(**table_scan_config.model_dump(exclude_none=True))
    tasks = list(scan.plan_files())
    logger.info(f"Files to scan: {len(tasks)}")

    output_columns = [
        field.name
        for field in INVENTORY_TABLE_SCHEMA
        if field.name in table_scan_config.selected_fields
    ]
    output_schema = pyarrow.schema(
        [
            f
            for f in INVENTORY_TABLE_SCHEMA
            if f.name in table_scan_config.selected_fields
        ]
    )

    with pyarrow.parquet.ParquetWriter(
        where=inventory_parquet_path,
        schema=output_schema,
        compression="zstd",
    ) as writer:
        for i, task in enumerate(tasks):
            if task.delete_files or task.residual != AlwaysTrue():
                # Fall back to PyIceberg's own reader for tasks with deletes or
                # residual filters — direct reads would miss deleted/filtered rows.
                logger.warning(
                    f"Task {i} has delete files or residual filter — using PyIceberg reader"
                )
                batch = table.scan(
                    **table_scan_config.model_dump(exclude_none=True)
                ).to_arrow()
                writer.write_table(batch.select(output_columns))
                continue

            batch = pyarrow.parquet.read_table(
                source=task.file.file_path,
                columns=output_columns,
            )
            writer.write_table(batch.cast(output_schema))
            logger.info(
                f"[{i + 1}/{len(tasks)}] {task.file.file_path} "
                f"— {len(batch):,} rows, {task.file.file_size_in_bytes / 2**20:.1f} MB"
            )


_DEFAULT_TABLE_CONFIG = IcebergTableConfig(
    catalog_config=S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
    ),
    namespace="b_imos-data",
    table_name="inventory",
)


@prefect.task
def extract(
    table_config: IcebergTableConfig = _DEFAULT_TABLE_CONFIG,
    table_scan_config: TableScanConfig = TableScanConfig(),
    inventory_parquet_path: pathlib.Path = pathlib.Path("imos-data.inventory.parquet"),
) -> polars.LazyFrame:
    table = table_config.load()
    sink_table(
        table=table,
        table_scan_config=table_scan_config,
        inventory_parquet_path=inventory_parquet_path,
    )
    return polars.scan_parquet(inventory_parquet_path)


if __name__ == "__main__":
    extract()
