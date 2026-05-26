"""
Prefect flow entry point: orchestrate on Fargate with a pre-built image.

Usage:
    uv run cluster-fargate-flow

Wraps the full pipeline in a top-level Prefect flow so it can be
deployed to Prefect Cloud/Server, scheduled, and monitored end-to-end.
The Docker image must be built and pushed to ECR before invoking this flow
(see run_fargate.py for the build/push step).
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

# Freeze already-installed packages as overrides so uv won't upgrade or
# replace them — only missing dependencies of data_index will be installed.
with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as _f:
    _freeze = subprocess.run(
        ["uv", "pip", "freeze", "--system"], capture_output=True, text=True, check=True
    )
    _f.write(_freeze.stdout)
    _f.flush()
    _result = subprocess.run(
        ["uv", "pip", "install", "--system", "--override", _f.name, "."],
        capture_output=True,
        text=True,
    )
    if _result.returncode != 0:
        raise RuntimeError(
            f"uv pip install failed (exit {_result.returncode}):\n"
            f"{_result.stdout}\n{_result.stderr}"
        )

import prefect  # noqa: E402
import prefect_dask  # noqa: E402

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner  # noqa: E402
from data_index.cluster.fargate_cluster_config import (
    PrefectFargateClusterConfig,  # noqa: E402
)
from data_index.cluster.orchestrate import orchestrate  # noqa: E402
from data_index.file_fetcher import S5CMDFetcher  # noqa: E402
from data_index.iceberg_config import S3TablesCatalogConfig  # noqa: E402
from data_index.iceberg_config.iceberg_table_config import (
    IcebergTableConfig,  # noqa: E402
)
from data_index.inventory_source.live_s3 import LiveS3InventorySource  # noqa: E402
from data_index.metadata_extractor import UnstructuedNetCDFExtractor  # noqa: E402
from data_index.s3_metadata.extract import TableScanConfig  # noqa: E402
from data_index.structured_sink import StructuredS3TableSink  # noqa: E402
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata  # noqa: E402
from data_index.unstructured_sink import UnstructuredS3TableSink  # noqa: E402

ECR_REGISTRY = "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com"
DEFAULT_IMAGE = f"{ECR_REGISTRY}/prefect:prefect-dask"
REGION = "ap-southeast-2"
CATALOG_ARN = "arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3"
BATCH_SIZE = 1_000
THRESHOLD_BYTES = 10 * 1024**2  # 10 MB


@prefect.flow
def cloud_fargate(
    prefect_fargate_cluster_config: PrefectFargateClusterConfig = PrefectFargateClusterConfig(
        n_workers=4,
        image=DEFAULT_IMAGE,
        cpu_architecture="ARM64",
        scheduler_cpu=4096,
        scheduler_mem=16384,
        worker_cpu=4096,
        worker_mem=16384,
    ),
    batch_size: int = BATCH_SIZE,
) -> None:
    """
    Top-level Prefect flow for a full Fargate-backed data-index run.

    Args:
        image: Fully-qualified Docker image URI for the Fargate workers.
               Must already be pushed to ECR before this flow runs.
        n_workers: Initial Fargate worker count for the Dask cluster.
        worker_cpu: vCPU in milli-units per worker (4096 = 4 vCPU).
        worker_mem: Memory in MB per worker.
        batch_size: Maximum files per Batch dispatched to a worker.
    """

    if prefect_fargate_cluster_config is None:
        prefect_fargate_cluster_config = PrefectFargateClusterConfig(
            n_workers=4,
            image=DEFAULT_IMAGE,
            cpu_architecture="ARM64",
            scheduler_cpu=4096,
            scheduler_mem=16384,
            worker_cpu=4096,
            worker_mem=16384,
        )

    logger = prefect.get_run_logger()

    logger.info(f"Using image: {prefect_fargate_cluster_config.image}")

    catalog_config = S3TablesCatalogConfig(region=REGION, arn=CATALOG_ARN)
    catalog = catalog_config.build()

    orchestrate.with_options(
        task_runner=prefect_dask.DaskTaskRunner(
            cluster_class="dask_cloudprovider.aws.FargateCluster",
            cluster_kwargs=prefect_fargate_cluster_config.model_dump(exclude_none=True),
        ),
    )(
        inventory_source=LiveS3InventorySource(
            table_config=IcebergTableConfig(
                catalog_config=catalog_config,
                namespace="b_imos-data",
                table_name="inventory",
            ),
            table_scan_config=TableScanConfig(row_filter="key LIKE 'IMOS/%'"),
            path=pathlib.Path(".extract/s3_metadata"),
            skip_if_exists=True,
        ),
        partitioner=GreedyBatchPartitioner(
            max_files=batch_size,
            max_bytes=50 * 1024**3,
        ),
        fetcher=S5CMDFetcher(anon=True),
        extractor=UnstructuedNetCDFExtractor(),
        structured_sink=StructuredS3TableSink(
            catalog=catalog,
            namespace="structured-metadata",
            table_name="index",
        ),
        unstructured_sink=UnstructuredS3TableSink(
            catalog=catalog,
            namespace="unstructured-metadata",
            table_name="index",
        ),
        metadata_factory=InMemoryUnstructuredMetadata,
    )
