import typing

import pydantic

import data_index.protocols
import data_index.schema.metadata


class DummySink(pydantic.BaseModel):
    type: typing.Literal["dummy_sink"] = pydantic.Field(default="dummy_sink")

    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self,
        metadata: list[data_index.schema.metadata.StructuredMetadata]
        | list[data_index.schema.metadata.UnstructuredMetadata]
        | list[data_index.protocols.DeadLetter],
    ) -> None:
        """Persist data"""
        ...
