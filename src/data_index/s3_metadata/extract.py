import contextlib
import pathlib

import polars
import prefect
import pyarrow
import pyarrow.parquet
import pyiceberg.table

from data_index.iceberg_config import S3TablesCatalogConfig
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig

from ._schema import INVENTORY_TABLE_SCHEMA

# Alias for backward compatibility
TableScanConfig = IcebergTableScanConfig


@contextlib.contextmanager
def constrained_arrow_threads(
    target_cpu_count: int = 1,
    target_io_thread_count: int = 1,
):
    """
    Thread and locking resource contention issues make attempting
    an serial thread pool approach necessary
    """
    original_cpu_count = pyarrow.cpu_count()
    original_io_thread_count = pyarrow.io_thread_count()
    try:
        pyarrow.set_cpu_count(target_cpu_count)
        pyarrow.set_io_thread_count(target_io_thread_count)
        yield
    finally:
        pyarrow.set_cpu_count(original_cpu_count)
        pyarrow.set_io_thread_count(original_io_thread_count)


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
                logger.info(f"Attempting to write batch: {len(batch)} rows")
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
