import pydantic
import xarray
import pathlib

class DiskXarrayHandle(pydantic.BaseModel):
    """
    Cache the necessary information to resolve a disk file to an `xarray.Dataset`
    """
    path: pathlib.Path
    s3_uri: str

    @property
    def ds(self) -> xarray.Dataset:
        return xarray.open_dataset(filename_or_obj=self.path)

    def cleanup(self) -> None:
        if self.path.exists():
            self.path.unlink()