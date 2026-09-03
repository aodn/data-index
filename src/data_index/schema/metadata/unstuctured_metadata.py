import dataclasses
import typing

from .base_metadata import BaseMetadata


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class UnstructuredMetadata(BaseMetadata):
    """Unstructured metadata row schema and backend schema converters.

    `UnstructuredMetadata` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    # Upgrade the schema version when changing the schema
    SCHEMA_VERSION: typing.ClassVar[int] = 4
    schema_version: int = SCHEMA_VERSION

    metadata: str
