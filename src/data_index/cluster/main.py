import logging
import pathlib

import docker_image
import fargate_cluster_config
import prefect
import prefect_dask
import rich
import sh

logging.basicConfig(level=logging.INFO)

# Log Docker into ECR
password = sh.aws("ecr", "get-login-password", "--region", "ap-southeast-2")
sh.docker(
    "login", 
    "--username", "AWS", 
    "--password-stdin", 
    "704910415367.dkr.ecr.ap-southeast-2.amazonaws.com",
    _in=password,
)

# Set up a docker image
docker_image_ = docker_image.DockerImage(
    name="704910415367.dkr.ecr.ap-southeast-2.amazonaws.com/prefect",
    tag="prefect-dask",
    dockerfile=pathlib.Path("Dockerfile"),
)
prefect_docker_image = docker_image_.PrefectDockerImage

# Build and push docker image
# Get's pushed to ecr via the name; bit weird to me but hey it works
# Note it builds by default on the hosts architecture... if on modern mac that is arm64 not x86_64!
prefect_docker_image.build()
prefect_docker_image.push()

# Set up some config for the fargate cluster
fargate_cluster_config = fargate_cluster_config.PrefectFargateClusterConfig(
    n_workers=1,
    image=f"{docker_image_.name}:{docker_image_.tag}",
    cpu_architecture="ARM64",
    scheduler_cpu=1024,
    scheduler_mem=2048,
    worker_cpu=1024,
    worker_mem=2048,
)
rich.print(fargate_cluster_config.model_dump(exclude_none=True))

@prefect.task
def other_task():
    pass

# Dask worker item
@prefect.task
def sq(x: int) -> int:
    other_task()
    return x**2

# Dask Scheduler, essentially (I think?), distributing the work

# AI GENERATED

# In a Prefect + Dask setup, the roles actually look like this:

# The Flow: This is the Client. It runs the Python code that says "Hey Dask, here is a list of tasks I need done."

# The Dask Scheduler: This is a separate process (the "brain") that lives inside your Fargate cluster. It receives the tasks from the Flow and decides which worker is free.

# The Dask Workers: These are the "muscles" (the Fargate containers). They execute the tasks.

# The Prefect Tasks: These are the Units of Work (the functions) being sent to the workers.
@prefect.flow(
    task_runner=prefect_dask.DaskTaskRunner(
        # This is where the temporary cluster get's tied to the flow
        cluster_class="dask_cloudprovider.aws.FargateCluster",
        cluster_kwargs=fargate_cluster_config.model_dump(exclude_none=True),
    ),
)
def fargate_flow(xs: list[int]) -> list[int]:
    result = sq.map(xs)
    return result.result()


if __name__ == "__main__":
    fargate_flow(xs=[0, 1, 2, 3, 4, 5, 6, 7])
