from __future__ import annotations

import typing

import pydantic


class PrefectFargateClusterConfig(pydantic.BaseModel):
    """
    Configuration for Dask ECS Clusters running on AWS Fargate.
    Validated against AWS Fargate's architectural constraints.
    """

    model_config = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",  # Prevents passing invalid args like 'worker_gpu'
    )

    # --- Fargate Launch Configuration ---
    cpu_architecture: typing.Literal["x86_64", "ARM64"]
    # TODO: Consider removing scheduler and workers flags
    # By default when passing configuration into prefect, the scheduler and workers are set to fargate
    # fargate_scheduler: bool = pydantic.Field(
    #     default=True, description="Enable Fargate launch type for the scheduler."
    # )
    # fargate_workers: bool = pydantic.Field(
    #     default=True, description="Enable Fargate launch type for workers."
    # )
    fargate_spot: bool = pydantic.Field(
        default=True, description="Use Fargate Spot capacity providers for workers."
    )

    # --- Resource Allocation ---
    image: str = pydantic.Field(
        description="Docker image for scheduler and workers.",
    )
    scheduler_cpu: int = pydantic.Field(
        default=1024,
        ge=256,
        le=16384,
        description="vCPU in milli-units (1024 = 1 vCPU)",
    )
    scheduler_mem: int = pydantic.Field(
        default=4096,
        ge=512,
        le=122880,
        description="Memory in Megabytes",
    )

    worker_cpu: int = pydantic.Field(default=4096)
    worker_mem: int = pydantic.Field(default=16384)
    worker_nthreads: int = pydantic.Field(default=1)
    n_workers: int | None = pydantic.Field(
        default=None, gt=0, description="Initial worker count."
    )

    # TODO: Understand this
    # For now switching off because I have no idea what most of this does...
    # --- Networking & IAM ---
    # vpc: str | None = pydantic.Field(
    #     default=None, description="VPC ID (e.g., vpc-12345)."
    # )
    # subnets: list[str] | None = pydantic.Field(
    #     default=None, description="List of subnet IDs."
    # )
    # security_groups: list[str] | None = pydantic.Field(
    #     default=None, description="Existing Security Group IDs."
    # )
    # fargate_use_private_ip: bool = pydantic.Field(
    #     default=False, description="Use private IPs for inter-cluster communication."
    # )

    execution_role_arn: str | None = pydantic.Field(
        default=None, description="Role for ECS agent permissions."
    )
    task_role_arn: str | None = pydantic.Field(
        default=None, description="Role for application permissions."
    )

    # --- Runtime Settings ---
    environment: dict[str, str] = pydantic.Field(
        default_factory=dict, description="Environment variables for the containers."
    )
    tags: dict[str, str] = pydantic.Field(
        default_factory=dict, description="AWS Tags applied to all resources."
    )
    skip_cleanup: bool = pydantic.Field(
        default=False,
        description="Skip cleaning up of stale resources. Useful if you have lots of resources and this operation takes awhile.",
    )

    # TODO: Needs re-work against actual fargate constraints; these are hallucinated nonsense
    # @pydantic.model_validator(mode="after")
    # def check_fargate_ratios(self) -> FargateClusterConfig:
    #     """
    #     Validates basic Fargate CPU/Memory compatibility.
    #     Example: 1024 (1 vCPU) must be paired with memory between 2GB and 8GB.
    #     """
    #     # Logic for scheduler
    #     if self.scheduler_cpu == 1024 and not (2048 <= self.scheduler_mem <= 8192):
    #         raise ValueError(
    #             f"Invalid scheduler memory ({self.scheduler_mem}MB) for 1024 CPU units."
    #         )

    #     # Logic for worker
    #     if self.worker_cpu == 4096 and not (8192 <= self.worker_mem <= 30720):
    #         raise ValueError(
    #             f"Invalid worker memory ({self.worker_mem}MB) for 4096 CPU units."
    #         )

    #     return self
