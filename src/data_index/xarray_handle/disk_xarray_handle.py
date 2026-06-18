import pathlib

import pydantic
import xarray

from data_index.xarray_handle._magic import format_from_magic


class DiskXarrayHandle(pydantic.BaseModel):
    """
    Cache the necessary information to resolve a disk file to an `xarray.Dataset`
    """

    path: pathlib.Path
    s3_uri: str
    _dataset: xarray.Dataset | None = pydantic.PrivateAttr(default=None)

    @property
    def file_format(self) -> str | None:
        with open(self.path, "rb") as f:
            return format_from_magic(f.read(8))

    @property
    def ds(self) -> xarray.Dataset:
        if self._dataset is None:
            self._dataset = xarray.open_dataset(filename_or_obj=self.path)
        return self._dataset

    def cleanup(self) -> None:
        if self.path.exists():
            self.path.unlink()
