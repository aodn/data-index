import pathlib

import prefect
import prefect_dask

from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config import (
    IcebergTableConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import LiveS3InventorySource, ParquetInventorySource
from data_index.inventory_source.live_s3_facility_subset import (
    LiveS3InventorySourceFacilitySubset,
)
from data_index.structured_metadata import StructuredMetadata
from data_index.structured_sink import StructuredS3TableSink
from data_index.unstructured_sink import (
    UnstructuredS3TableSink,
)

from .local import (
    S5CMD_WORKERS,
    TRANSFORM_WORKERS,
    extractor,
    inventory_table_config,
    inventory_table_scan_config,
    partitioner,
    run_index_local,
)

live_inventory_source = LiveS3InventorySourceFacilitySubset(
    table_config=inventory_table_config,
    table_scan_config=inventory_table_scan_config,
    path=pathlib.Path(".extract/s3_metadata"),
    skip_if_exists=True,
    subset_per_facility=2_000,
)

# --- Sink config ---
data_index_catalog_config = S3TablesCatalogConfig(
    region="ap-southeast-2",
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)

structured_metadata_table_config = IcebergTableConfig(
    catalog_config=data_index_catalog_config,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

structured_sink = StructuredS3TableSink(
    iceberg_table_config=structured_metadata_table_config,
)

unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=data_index_catalog_config,
    namespace="data_index",
    table_name="unstructured_metadata",
)

unstructured_sink = UnstructuredS3TableSink(
    iceberg_table_config=unstructured_metadata_table_config,
)

# --- Fetcher ---
size_threshold_bytes = 1024**2 * 10
fetcher = ThresholdFileFetcher(
    size_threshold_bytes=size_threshold_bytes,
    disk_fetcher=S5CMDFetcher(num_workers=S5CMD_WORKERS, anon=True),
    cloud_fetcher=S3Fetcher(block_size=size_threshold_bytes),
)


@prefect.flow
def run_index_local_subset(
    inventory_source: LiveS3InventorySource
    | ParquetInventorySource = live_inventory_source,
    partitioner=partitioner,
    fetcher=fetcher,
    extractor=extractor,
    structured_sink=structured_sink,
    unstructured_sink=unstructured_sink,
    metadata_factory=None,
    transform_max_workers: int | None = TRANSFORM_WORKERS,
):
    run_index_local.with_options(task_runner=prefect_dask.DaskTaskRunner())(
        inventory_source=inventory_source,
        partitioner=partitioner,
        fetcher=fetcher,
        extractor=extractor,
        structured_sink=structured_sink,
        unstructured_sink=unstructured_sink,
        metadata_factory=metadata_factory,
        transform_max_workers=transform_max_workers,
    )


if __name__ == "__main__":
    run_index_local_subset()
