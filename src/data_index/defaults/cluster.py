import pathlib

import prefect
import prefect_dask

import data_index
from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.cluster import DockerImage, PrefectFargateClusterConfig
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config import (
    IcebergTableConfig,
    IcebergTableScanConfig,
    S3TablesCatalogConfig,
)
from data_index.inventory_source import LiveS3InventorySource, ParquetInventorySource
from data_index.metadata_extractor import NetCDFExtractor, UnstructuedNetCDFExtractor
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


# --- Live Inventory Source config
_s3_metadata_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
)

_inventory_table_config = IcebergTableConfig(
    catalog_config=_s3_metadata_catalog_config,
    namespace="b_imos-data",
    table_name="inventory",
)

_inventory_table_scan_config = IcebergTableScanConfig(row_filter="key LIKE 'IMOS/%'")

_live_inventory_source = LiveS3InventorySource(
    table_config=_inventory_table_config,
    table_scan_config=_inventory_table_scan_config,
    path=pathlib.Path(".extract/s3_metadata"),
    skip_if_exists=True,
)

# --- Static Inventory Source config ---
_static_inventory_source = ParquetInventorySource(
    path="s3://aodn-dataflow-dev/thomas.galindo/processing/stored/s3_metadata/"
)


# --- Partitioner config ---
_greedy_partitioner = GreedyBatchPartitioner(
    max_files=BATCH_SIZE,
    max_bytes=50 * 1024**3,
)

# --- File fetcher ---
_file_fetcher = S5CMDFetcher(anon=True)

# --- Metadata extractor ---
_unstructured_netcdf_extractor = UnstructuedNetCDFExtractor()

# --- Sink config ---
_data_index_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/data-index",
)

_structured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data-index",
    table_name="structured-metadata",
)

_structured_s3_table_sink = StructuredS3TableSink(
    iceberg_table_config=_structured_metadata_table_config,
)

_unstructured_metadata_table_config = IcebergTableConfig(
    catalog_config=_data_index_catalog_config,
    namespace="data-index",
    table_name="unstructured-metadata",
)

_unstructured_s3_table_sink = UnstructuredS3TableSink(
    iceberg_table_config=_unstructured_metadata_table_config,
)

# --- Docker image config ---
docker_image = DockerImage(
    name=f"{ECR_REGISTRY}/prefect",
    tag="prefect-dask",
    dockerfile=pathlib.Path("src/data_index/cluster/Dockerfile"),
)

# --- Fargate Cluster config ---
_fargate_cluster_options = PrefectFargateClusterConfig(
    n_workers=4,
    image=docker_image.full_name,
    cpu_architecture="ARM64",
    scheduler_cpu=1024,
    scheduler_mem=2048,
    worker_cpu=4096,
    worker_mem=16384,
)


@prefect.flow
def run_index_cluster(
    inventory_source: LiveS3InventorySource
    | ParquetInventorySource = _live_inventory_source,
    partitioner: GreedyBatchPartitioner = _greedy_partitioner,
    fetcher: S3Fetcher | S5CMDFetcher | ThresholdFileFetcher = _file_fetcher,
    extractor: NetCDFExtractor
    | UnstructuedNetCDFExtractor = _unstructured_netcdf_extractor,
    structured_sink: StructuredParquetSink
    | StructuredS3TableSink = _structured_s3_table_sink,
    unstructured_sink: UnstructuredParquetSink
    | UnstructuredS3TableSink = _unstructured_s3_table_sink,
    metadata_factory: InMemoryUnstructuredMetadata
    | DiskCachedUnstructuredMetadata
    | None = None,
    transform_max_workers: int | None = None,
    fargate_cluster_options: PrefectFargateClusterConfig = _fargate_cluster_options,
):

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


if __name__ == "__main__":
    run_index_cluster()
