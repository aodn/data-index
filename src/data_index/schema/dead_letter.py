import dataclasses
import typing

import prefect.runtime.flow_run

import data_index.protocols

from .schema import Schema


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class DeadLetter(data_index.protocols.ObjectReference, Schema):
    """
    DeadLetter row schema and backend schema converters.

    `DeadLetter` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    SCHEMA_VERSION: typing.ClassVar[int] = 1
    schema_version: int = SCHEMA_VERSION

    error: str | None
    index_flow_id: str | None = dataclasses.field(
        default_factory=lambda: prefect.runtime.flow_run.get_parent_flow_run_id()
    )
    batch_flow_id: str | None = dataclasses.field(
        default_factory=lambda: prefect.runtime.flow_run.get_id()
    )

    @classmethod
    def from_object_reference(
        cls,
        object_reference: data_index.protocols.ObjectReference,
        error: str | None,
    ) -> typing.Self:
        return cls(
            bucket=object_reference.bucket,
            key=object_reference.key,
            version_id=object_reference.version_id,
            size=object_reference.size,
            error=error,
        )


if __name__ == "__main__":
    object_reference = data_index.protocols.ObjectReference(
        bucket="test",
        key="a.nc",
        version_id="0",
        size=32,
    )
    dead_letter = DeadLetter.from_object_reference(
        object_reference=object_reference,
        error="test",
    )
    import rich

    rich.print(typing.get_type_hints(DeadLetter, include_extras=True))
    rich.print(object_reference)
    rich.print(dead_letter)
    rich.print(dead_letter.as_pyarrow_schema())

    # Should be possible to get the flow run ids easily
    #
    # https://docs.prefect.io/v3/api-ref/python/prefect-runtime-flow_run
    #
    # from prefect.runtime import flow_run
    #
    # # Inside any flow or task:
    # current_id = flow_run.id
    # parent_id = flow_run.parent_flow_run_id  # None if not a subflow
    #
    # Should just be an append mechanism. No need to upsert.
    #
    # How to retry the Dead Letter Queue?
    # Write an Inventory Source for the Dead Letter Queue!
