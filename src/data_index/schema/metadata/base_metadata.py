import dataclasses
import typing

import pyarrow

import data_index.schema


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class BaseMetadata(data_index.schema.Schema):
    """Metadata row schema to inherit from"""

    bucket: str
    key: str
    version_id: str
    hash: str
    file_format: str
    facility: str

    @classmethod
    def to_arrow(
        cls,
        metadata: list[typing.Self],
    ) -> pyarrow.Table:
        """Convert a list of self to a validated pyarrow table"""
        return pyarrow.Table.from_pylist(
            [dataclasses.asdict(obj=metadata) for metadata in metadata],
            schema=cls.as_pyarrow_schema(),
        )
