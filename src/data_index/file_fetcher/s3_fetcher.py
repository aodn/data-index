import cloudpathlib

from data_index.protocols import BatchEntry, XarrayHandle
from data_index.xarray_handle import S3XarrayHandle
import pydantic


class S3Fetcher(pydantic.BaseModel):
    """
    FileFetcher implementation that downloads files from S3 using boto3.

    Note this fetcher does not actually load any data.

    It passes back handles that intelligently query header information from NetCDF files in Cloud.
    """

    block_size: int = pydantic.Field(default=1024**2)

    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:
        return [
            S3XarrayHandle(
                path=cloudpathlib.S3Path(cloud_path=entry.uri),
                block_size=self.block_size,
            )
            for entry in entries
        ]
