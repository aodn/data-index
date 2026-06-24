from typing import Literal

import pydantic
from prefect.task_runners import ProcessPoolTaskRunner, ThreadPoolTaskRunner


class ThreadPoolRunnerConfig(pydantic.BaseModel):
    """Configuration specific to ThreadPoolTaskRunner."""

    type: Literal["thread_pool"] = pydantic.Field(default="thread_pool")
    max_workers: int = pydantic.Field(default=4, ge=1, le=64)

    def create(self) -> ThreadPoolTaskRunner:
        return ThreadPoolTaskRunner(
            max_workers=self.max_workers,
        )


class ProcessPoolRunnerConfig(pydantic.BaseModel):
    """Configuration specific to ProcessPoolTaskRunner."""

    type: Literal["process_pool"] = pydantic.Field(default="process_pool")
    max_workers: int = pydantic.Field(default=4, ge=1, le=64)

    def create(self) -> ProcessPoolTaskRunner:
        return ProcessPoolTaskRunner(
            max_workers=self.max_workers,
        )
