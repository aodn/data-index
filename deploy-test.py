import prefect
import prefect_github

from data_index.cluster.docker_image import DockerImage
from data_index.cluster.fargate_cluster_config import PrefectFargateClusterConfig
from data_index.cluster.orchestrate import orchestrate

# --- Docker images ---

deployment_docker_image = DockerImage(
    name="data-index",
    tag="latest",
    dockerfile="Dockerfile",
)
cluster_docker_image = DockerImage(
    name="data-index",
    tag="latest",
    dockerfile="Dockerfile",
)

# --- Fargate cluster config ---
fargate_config = PrefectFargateClusterConfig(
    n_workers=4,
    image=cluster_docker_image.full_name,
    cpu_architecture="ARM64",
    scheduler_cpu=1024,
    scheduler_mem=2048,
    worker_cpu=4096,
    worker_mem=16384,
)


@prefect.flow
def data_index_blank_implementation(
    # inventory_source: LiveS3InventorySource,
    # partitioner: GreedyBatchPartitioner,
    # fetcher: S5CMDFetcher | S3Fetcher | ThresholdFileFetcher,
    # extractor: NetCDFExtractor | UnstructuedNetCDFExtractor,
    # structured_sink: StructuredS3TableSink,
    # unstructured_sink: UnstructuredS3TableSink,
) -> None:

    orchestrate(
        # inventory_source=inventory_source,
        # partitioner=partitioner,
        # fetcher=fetcher,
        # extractor=extractor,
        # structured_sink=structured_sink,
        # unstructured_sink=unstructured_sink,
        # metadata_factory=InMemoryUnstructuredMetadata,
    )


github_repo = prefect_github.GitHubRepository.load("data-index")
prefect.flow.from_source(
    source=github_repo, entrypoint="deploy-test.py:data_index_blank_implementation"
).deploy(
    name="implementation_deployment",
    work_pool_name="docker",
    image=deployment_docker_image.PrefectDockerImage,
    push=False,
    build=False,
)
