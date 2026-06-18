import pathlib

import prefect
import prefect_dask

import data_index
from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.cluster import DockerImage, PrefectFargateClusterConfig
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config import (
    IcebergTableConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import (
    LiveS3InventorySource,
    ParquetInventorySource,
    S3TableFacilitySubsetInventorySource,
    S3TableInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
    NetCDFExtractor,
    UnstructuedNetCDFExtractor,
)
from data_index.structured_metadata import StructuredMetadata
from data_index.structured_sink import StructuredParquetSink, StructuredS3TableSink
from data_index.unstructured_metadata import (
    DiskCachedUnstructuredMetadata,
    InMemoryUnstructuredMetadata,
)
from data_index.unstructured_sink import (
    UnstructuredParquetSink,
    UnstructuredS3TableSink,
)

# --- General config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
BATCH_SIZE = 1_000
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB
S5CMD_WORKERS = 8
TRANSFORM_MAX_WORKERS = 32


# --- Live Inventory Source config
_s3_metadata_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
)

_inventory_table_config = IcebergTableConfig(
    catalog_config=_s3_metadata_catalog_config,
    namespace="inventory",
    table_name="live",
)

_live_inventory_source = S3TableFacilitySubsetInventorySource(
    table_config=_inventory_table_config,
    subset_per_facility=10_000,
)

# --- Partitioner config ---
_greedy_partitioner = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=10 * 1024**3,
)

# --- File fetcher ---
_file_fetcher = ThresholdFileFetcher(
    size_threshold_bytes=THRESHOLD_BYTES,
    disk_fetcher=S5CMDFetcher(num_workers=S5CMD_WORKERS, anon=True),
    cloud_fetcher=S3Fetcher(block_size=THRESHOLD_BYTES),
)

# --- Metadata extractor ---
_attribute_netcdf_extractor = AttributeNetCDFExtractor()

# --- Sink config ---
_data_index_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)

_structured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

_structured_s3_table_sink = StructuredS3TableSink(
    iceberg_table_config=_structured_metadata_table_config,
)

_unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data_index",
    table_name="unstructured_metadata",
)

_unstructured_s3_table_sink = UnstructuredS3TableSink(
    iceberg_table_config=_unstructured_metadata_table_config,
)

# --- Docker image config ---
docker_image = DockerImage(
    name=f"{ECR_REGISTRY}/prefect",
    tag="prefect-dask",
    dockerfile=pathlib.Path("Dockerfile"),
)

# --- Fargate Cluster config ---
_fargate_cluster_options = PrefectFargateClusterConfig(
    n_workers=8,
    image=docker_image.full_name,
    cpu_architecture="ARM64",
    scheduler_cpu=4096,
    scheduler_mem=16384,
    worker_cpu=4096,
    worker_mem=16384,
    execution_role_arn="arn:aws:iam::704910415367:role/prefect-dask-execution",
    task_role_arn="arn:aws:iam::704910415367:role/prefect-dask-task",
)


@prefect.flow
def run_index_cluster(
    inventory_source: LiveS3InventorySource
    | ParquetInventorySource
    | S3TableInventorySource = _live_inventory_source,
    partitioner: GreedyBatchPartitioner = _greedy_partitioner,
    fetcher: S3Fetcher | S5CMDFetcher | ThresholdFileFetcher = _file_fetcher,
    extractor: AttributeNetCDFExtractor
    | NetCDFExtractor
    | UnstructuedNetCDFExtractor = _attribute_netcdf_extractor,
    structured_sink: StructuredParquetSink
    | StructuredS3TableSink = _structured_s3_table_sink,
    unstructured_sink: UnstructuredParquetSink
    | UnstructuredS3TableSink = _unstructured_s3_table_sink,
    metadata_factory: InMemoryUnstructuredMetadata
    | DiskCachedUnstructuredMetadata
    | None = None,
    transform_max_workers: int | None = TRANSFORM_MAX_WORKERS,
    fargate_cluster_options: PrefectFargateClusterConfig = _fargate_cluster_options,
):

    # Create flow run specific tables for testing
    flow_run_id = prefect.runtime.flow_run.get_id()
    if flow_run_id is not None:
        flow_run_id = flow_run_id.replace("-", "_")
        if isinstance(structured_sink, StructuredS3TableSink):
            structured_sink.iceberg_table_config.table_name = (
                f"{structured_sink.iceberg_table_config.table_name}_{flow_run_id}"
            )
        if isinstance(unstructured_sink, UnstructuredS3TableSink):
            unstructured_sink.iceberg_table_config.table_name = (
                f"{unstructured_sink.iceberg_table_config.table_name}_{flow_run_id}"
            )

    # Run in try finally block to ensure cluster de-provision
    try:
        # Run with cluster options
        data_index.orchestrate.with_options(
            task_runner=prefect_dask.DaskTaskRunner(
                cluster_class="dask_cloudprovider.aws.FargateCluster",
                cluster_kwargs=fargate_cluster_options.model_dump(exclude_none=True),
            ),
        )(
            inventory_source,
            partitioner,
            fetcher,
            extractor,
            structured_sink,
            unstructured_sink,
            metadata_factory,
            transform_max_workers,
        )
    finally:
        ...


if __name__ == "__main__":
    # # Upload the task image
    # prefect_docker_image = docker_image.PrefectDockerImage
    # prefect_docker_image.build()
    # prefect_docker_image.push()
    run_index_cluster.serve(
        name="run-index-cluster",
    )
