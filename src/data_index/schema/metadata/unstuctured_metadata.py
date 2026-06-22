import typing

from .base import Metadata


class UnstructuredMetadata(Metadata):
    """Structured metadata row schema and backend schema converters.

    ``StructuredMetadata`` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    # Upgrade the schema version when changing the schema
    SCHEMA_VERSION: typing.ClassVar[int] = 2
    schema_version: int = SCHEMA_VERSION

    metadata = str
