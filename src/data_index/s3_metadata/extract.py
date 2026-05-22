import pathlib

import polars
import prefect
import pyarrow.parquet
import pyiceberg.expressions
import pyiceberg.table
import pydantic

from data_index.iceberg_config import S3TablesCatalogConfig
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from ._schema import INVENTORY_TABLE_SCHEMA


class TableScanConfig(pydantic.BaseModel):
    """Configuration for an Iceberg table scan."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    row_filter: str | pyiceberg.expressions.BooleanExpression | None = pydantic.Field(
        default=None, description="Filter expression for rows to include in the scan."
    )
    selected_fields: tuple[str, ...] = pydantic.Field(
        default=(
            "bucket",
            "key",
            "version_id",
            "size",
            "sequence_number",
            "is_delete_marker",
        ),
        description="List of columns to project.",
    )
    case_sensitive: bool = pydantic.Field(
        default=True,
        description="Whether column name lookups should be case sensitive.",
    )
    snapshot_id: int | None = pydantic.Field(
        default=None,
        description="The ID of the snapshot to read for time-travel queries.",
    )
    limit: int | None = pydantic.Field(
        default=None, description="Maximum number of rows to return."
    )


@prefect.task
def sink_table(
    table: pyiceberg.table.Table,
    table_scan_config: TableScanConfig,
    inventory_parquet_path: pathlib.Path,
):
    logger = prefect.get_run_logger()

    inventory_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Table scan config:\n{table_scan_config.model_dump_json(indent=4)}")

    with table.scan(
        **table_scan_config.model_dump(exclude_none=True)
    ).to_arrow_batch_reader() as batches:
        with pyarrow.parquet.ParquetWriter(
            where=inventory_parquet_path,
            schema=pyarrow.schema(
                [
                    field
                    for field in INVENTORY_TABLE_SCHEMA
                    if field.name in table_scan_config.selected_fields
                ]
            ),
            compression="zstd",
        ) as writer:
            for batch in batches:
                writer.write_batch(batch=batch)
                logger.info(
                    f"Wrote batch of in memory size {batch.get_total_buffer_size()} bytes..."
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
