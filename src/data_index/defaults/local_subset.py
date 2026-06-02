import pathlib

import prefect

from data_index.inventory_source import LiveS3InventorySource, ParquetInventorySource
from data_index.inventory_source.live_s3_facility_subset import (
    LiveS3InventorySourceFacilitySubset,
)
from data_index.structured_sink import StructuredParquetSink
from data_index.unstructured_sink import UnstructuredParquetSink

from .local import (
    MAX_WORKERS,
    OUT_DIR,
    TRANSFORM_WORKERS,
    extractor,
    fetcher,
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
    subset_per_facility=10_000,
)

structured_sink = StructuredParquetSink(path=OUT_DIR / "structured_metadata.parquet")
unstructured_sink = UnstructuredParquetSink(
    path=OUT_DIR / "unstructured_metadata.parquet"
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
    run_index_local.with_options(
        task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=MAX_WORKERS)
    )(
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
