import typing

import pydantic

from data_index.protocols import ObjectReference
from data_index.xarray_handle import FSSpecXarrayHandle


class FSSpecFetcher(pydantic.BaseModel):
    """
    FileFetcher implementation that downloads files from S3 using boto3.

    Note this fetcher does not actually load any data.

    It passes back handles that intelligently query header information from NetCDF files in Cloud.
    """

    type: typing.Literal["s3_fetcher"] = pydantic.Field(default="s3_fetcher")

    block_size: int = pydantic.Field(default=1024**2)

    def fetch(self, object_references: list[ObjectReference]) -> list[ObjectReference]:
        return [
            object_reference.with_xarray_handle(
                xarray_handle=FSSpecXarrayHandle(
                    s3_uri=object_reference.as_versioned_uri(),
                )
            )
            for object_reference in object_references
        ]
