"""
Run the Orchestrator on AWS Fargate via a Dask cluster.

Usage (from repo root):
    uv run cluster-fargate

Steps:
  1. Logs into ECR and builds + pushes the Docker image
  2. Fetches the live inventory from the S3 Tables Iceberg catalog
  3. Runs the orchestrate flow with DaskTaskRunner → FargateCluster
"""

import logging
import pathlib

import prefect_dask
import rich
import sh

from data_index import orchestrate
from data_index.batch_partitioner.greedy import GreedyBatchPartitioner
from data_index.iceberg_config import S3TablesCatalogConfig
from data_index.cluster.docker_image import DockerImage
from data_index.cluster.fargate_cluster_config import PrefectFargateClusterConfig
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.inventory_source.live_s3 import LiveS3InventorySource
from data_index.metadata_extractor import UnstructuedNetCDFExtractor
from data_index.s3_metadata.extract import TableScanConfig
from data_index.structured_sink import StructuredS3TableSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.unstructured_sink import UnstructuredS3TableSink

logging.basicConfig(level=logging.INFO)

# --- Config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
BATCH_SIZE = 1_000
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Docker image ---
docker_image = DockerImage(
    name=f"{ECR_REGISTRY}/prefect",
    tag="prefect-dask",
    dockerfile=pathlib.Path("src/data_index/cluster/Dockerfile"),
)

# --- Fargate cluster config ---
fargate_config = PrefectFargateClusterConfig(
    n_workers=4,
    image=docker_image.full_name,
    cpu_architecture="ARM64",
    scheduler_cpu=1024,
    scheduler_mem=2048,
    worker_cpu=4096,
    worker_mem=16384,
)
rich.print(fargate_config.model_dump(exclude_none=True))

# --- Catalog config (AWS S3 Tables) ---
_catalog_config = S3TablesCatalogConfig(
    region=REGION,
    arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
)

# --- Inventory source ---
_inventory_table_config = IcebergTableConfig(
    catalog_config=_catalog_config,
    namespace="b_imos-data",
    table_name="inventory",
)

# --- Sink catalog (reuses same S3 Tables catalog) ---
_sink_table_config_structured = IcebergTableConfig(
    catalog_config=_catalog_config,
    namespace="structured-metadata",
    table_name="index",
)
_sink_table_config_unstructured = IcebergTableConfig(
    catalog_config=_catalog_config,
    namespace="unstructured-metadata",
    table_name="index",
)


def main() -> None:
    # Log Docker into ECR
    password = sh.aws("ecr", "get-login-password", "--region", REGION)
    sh.docker(
        "login",
        "--username",
        "AWS",
        "--password-stdin",
        ECR_REGISTRY,
        _in=password,
    )

    # Build and push image (note: build for ARM64 on Apple Silicon matches Fargate target)
    prefect_docker_image = docker_image.PrefectDockerImage
    prefect_docker_image.build()
    prefect_docker_image.push()

    sink_catalog = _catalog_config.build()

    orchestrate.with_options(
        task_runner=prefect_dask.DaskTaskRunner(
            cluster_class="dask_cloudprovider.aws.FargateCluster",
            cluster_kwargs=fargate_config.model_dump(exclude_none=True),
        ),
    )(
        inventory_source=LiveS3InventorySource(
            table_config=_inventory_table_config,
            table_scan_config=TableScanConfig(row_filter="key LIKE 'IMOS/%'"),
            path=pathlib.Path(".extract/s3_metadata"),
            skip_if_exists=True,
        ),
        partitioner=GreedyBatchPartitioner(
            max_files=BATCH_SIZE,
            max_bytes=50 * 1024**3,
        ),
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=THRESHOLD_BYTES,
            disk_fetcher=S5CMDFetcher(anon=True),
            cloud_fetcher=S3Fetcher(block_size=5 * 1024**2),
        ),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredS3TableSink(
            catalog=sink_catalog,
            namespace="structured-metadata",
            table_name="index",
        ),
        unstructured_sink=UnstructuredS3TableSink(
            catalog=sink_catalog,
            namespace="unstructured-metadata",
            table_name="index",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
    )


if __name__ == "__main__":
    main()
