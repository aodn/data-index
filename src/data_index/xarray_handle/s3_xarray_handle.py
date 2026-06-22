import cloudpathlib
import fsspec
import pydantic
import xarray

from data_index.protocols import ObjectReference
from data_index.xarray_handle._magic import format_from_magic


class S3XarrayHandle(pydantic.BaseModel):
    """
    Cache the necessary information to resolve a s3 file to an `xarray.Dataset`
    via fsspec
    """

    path: cloudpathlib.S3Path
    object_ref: ObjectReference
    cache_type: str = "blockcache"
    block_size: int = 1024**2
    _dataset: xarray.Dataset | None = pydantic.PrivateAttr(default=None)

    @property
    def s3_uri(self) -> str:
        return self.object_ref.as_uri()

    @property
    def file_format(self) -> str | None:
        s3fs = fsspec.filesystem(protocol="s3", anon=True)
        with s3fs.open(
            f"{self.object_ref.bucket}/{self.object_ref.key}",
            "rb",
            version_id=self.object_ref.version_id,
        ) as f:
            return format_from_magic(f.read(8))

    @property
    def ds(self) -> xarray.Dataset:
        if self._dataset is None:
            s3fs = fsspec.filesystem(protocol="s3", anon=True)
            file = s3fs.open(
                path=f"{self.object_ref.bucket}/{self.object_ref.key}",
                cache_type=self.cache_type,
                block_size=self.block_size,
                version_id=self.object_ref.version_id,
            )
            self._dataset = xarray.open_dataset(filename_or_obj=file)
        return self._dataset

    def cleanup(self) -> None:
        pass
