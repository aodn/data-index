import fsspec
import pydantic
import xarray

from data_index.xarray_handle._magic import format_from_magic


class FSSpecXarrayHandle(pydantic.BaseModel):
    """Cache the necessary information to resolve an S3 file to an `xarray.Dataset`

    safely and natively via fsspec with full versioning support.
    """

    s3_uri: str
    cache_type: str = "blockcache"
    block_size: int = 1024**2
    storage_options: dict[str, bool | str] = {"anon": True}

    _dataset: xarray.Dataset | None = pydantic.PrivateAttr(default=None)

    @property
    def file_format(self) -> str | None:
        try:
            fs = fsspec.filesystem("s3", **self.storage_options)
            # Opening via the versioned URI isolates this exact version
            with fs.open(self.s3_uri, "rb") as f:
                return format_from_magic(f.read(8))
        except Exception:
            return None

    @property
    def ds(self) -> xarray.Dataset:
        if self._dataset is None:
            fs = fsspec.filesystem("s3", **self.storage_options)

            # Open the file-like handle using fsspec's smart block caching mechanics
            file_obj = fs.open(
                path=self.s3_uri,
                cache_type=self.cache_type,
                block_size=self.block_size,
            )

            # Natively hand off to Xarray using the stable h5netcdf backend
            self._dataset = xarray.open_dataset(file_obj)

        return self._dataset

    def cleanup(self) -> None:
        """Safely close down the dataset to release network file handles and memory."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
