import dataclasses

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
