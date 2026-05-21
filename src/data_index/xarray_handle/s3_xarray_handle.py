import pydantic
import cloudpathlib
import xarray
import fsspec

class S3XarrayHandle(pydantic.BaseModel):
    """
    Cache the necessary information to resolve a s3 file to an `xarray.Dataset`
    via fsspec
    """
    path: cloudpathlib.S3Path
    cache_type: str = "blockcache"
    block_size: int = 1024 * 128

    @property
    def s3_uri(self) -> str:
        return self.path.as_uri()

    @property
    def ds(self) -> xarray.Dataset:
        s3fs = fsspec.filesystem(protocol="s3", anon=True)
        file = s3fs.open(
            path=f"{self.path.bucket}/{self.path.key}",
            cache_type=self.cache_type,
            block_size=self.block_size,
        )
        return xarray.open_dataset(filename_or_obj=file)

    def cleanup(self) -> None:
        pass