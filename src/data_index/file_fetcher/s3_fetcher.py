import cloudpathlib

from data_index.protocols import XarrayHandle
from data_index.xarray_handle import S3XarrayHandle

class S3Fetcher:
    """
    FileFetcher implementation that downloads files from S3 using boto3.
    
    Note this fetcher does not actually load any data.

    It passes back handles that intelligently query header information from NetCDF files in Cloud.
    """

    def fetch(self, uris: list[str]) -> list[XarrayHandle]:
        return [
            S3XarrayHandle(
                path=cloudpathlib.S3Path(cloud_path=uri),
            )
            for uri in uris
        ]
