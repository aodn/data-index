import dataclasses
import typing

from .base import Metadata


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class UnstructuredMetadata(Metadata):
    """Structured metadata row schema and backend schema converters.

    ``StructuredMetadata`` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    # Upgrade the schema version when changing the schema
    SCHEMA_VERSION: typing.ClassVar[int] = 3
    schema_version: int = SCHEMA_VERSION

    metadata: str
    file_format: str | None = None
    facility: str | None = None
