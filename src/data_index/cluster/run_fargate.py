"""
Run the Orchestrator on AWS Fargate via a Dask cluster.

Usage (from repo root):
    uv run python src/data_index/cluster/run_fargate.py

Steps:
  1. Logs into ECR and builds + pushes the Docker image
  2. Prepares a small inventory sample (or point INVENTORY_PATH at an existing one)
  3. Runs the orchestrate flow with DaskTaskRunner → FargateCluster
"""

import logging
import pathlib

import polars
import prefect_dask
import rich
import sh

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner
from data_index.cluster.docker_image import DockerImage
from data_index.cluster.fargate_cluster_config import PrefectFargateClusterConfig
from data_index.cluster.orchestrate import orchestrate
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.inventory_source.parquet import ParquetInventorySource
from data_index.metadata_extractor import UnstructuedNetCDFExtractor
from data_index.structured_sink import StructuredParquetSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.unstructured_sink import UnstructuredParquetSink

logging.basicConfig(level=logging.INFO)

# --- Config ---
ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
REGION = "ap-southeast-2"
LIMIT = 20
BATCH_SIZE = 5
OUT_DIR = pathlib.Path(".load/orchestrate-fargate")
INVENTORY_PATH = OUT_DIR / "inventory.parquet"
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


def prepare_inventory() -> None:
    """Sample LIMIT small .nc files from the local s3_metadata cache and write
    a prepared inventory (s3_uri, size) parquet for ParquetInventorySource."""
    df = (
        polars.scan_parquet(".extract/s3_metadata")
        .filter(
            polars.col("key").str.ends_with(".nc"),
            polars.col("size").le(1024**2),
        )
        .select(
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/",
            ).alias("s3_uri"),
            polars.col("size"),
        )
        .collect()
        .sample(LIMIT, seed=42)
    )
    df.write_parquet(INVENTORY_PATH)
    print(f"Prepared inventory: {len(df)} files → {INVENTORY_PATH}")


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

    prepare_inventory()

    orchestrate.with_options(
        task_runner=prefect_dask.DaskTaskRunner(
            cluster_class="dask_cloudprovider.aws.FargateCluster",
            cluster_kwargs=fargate_config.model_dump(exclude_none=True),
        ),
    )(
        inventory_source=ParquetInventorySource(path=INVENTORY_PATH),
        partitioner=GreedyBatchPartitioner(
            max_files=BATCH_SIZE,
            max_bytes=50 * 1024**3,
        ),
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=THRESHOLD_BYTES,
            disk_fetcher=S5CMDFetcher(),
            cloud_fetcher=S3Fetcher(block_size=5 * 1024**2),
        ),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredParquetSink(
            path=OUT_DIR / "structured_metadata.parquet",
        ),
        unstructured_sink=UnstructuredParquetSink(
            path=OUT_DIR / "unstructured_metadata.parquet",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
    )


if __name__ == "__main__":
    main()
