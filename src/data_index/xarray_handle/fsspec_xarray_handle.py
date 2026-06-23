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

            # Natively hand off to Xarray
            self._dataset = xarray.open_dataset(
                filename_or_obj=file_obj,
                decode_cf=False,
            )

        return self._dataset

    def cleanup(self) -> None:
        """Safely close down the dataset to release network file handles and memory."""
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None


## Potential refactor

# class FSSpecXarrayHandle(pydantic.BaseModel):
#     """Cache the necessary information to resolve an S3 file to an `xarray.Dataset`
#     safely and natively via fsspec with full versioning support.
#     """

#     s3_uri: str
#     cache_type: str = "blockcache"
#     block_size: int = 1024**2
#     storage_options: dict[str, Any] = {"anon": True}

#     # Track both the dataset and the file object privately
#     _file_obj: Any = pydantic.PrivateAttr(default=None)
#     _dataset: xarray.Dataset | None = pydantic.PrivateAttr(default=None)
#     _detected_format: str | None = pydantic.PrivateAttr(default=None)

#     def _init_file_obj(self) -> None:
#         """Ensure a single, cached file object is open."""
#         if self._file_obj is None:
#             fs = fsspec.filesystem("s3", **self.storage_options)
#             self._file_obj = fs.open(
#                 path=self.s3_uri,
#                 mode="rb",
#                 cache_type=self.cache_type,
#                 block_size=self.block_size,
#             )

#     @property
#     def file_format(self) -> str | None:
#         if self._detected_format is None:
#             try:
#                 self._init_file_obj()
#                 # Read from the cached file object. This populates the
#                 # first block of the cache so Xarray can reuse it!
#                 magic_bytes = self._file_obj.read(8)
#                 self._file_obj.seek(0)  # Rewind so Xarray can read from the start

#                 self._detected_format = format_from_magic(magic_bytes)
#             except Exception:
#                 return None
#         return self._detected_format

#     @property
#     def ds(self) -> xarray.Dataset:
#         if self._dataset is None:
#             self._init_file_obj()

#             # Map your custom format string to the actual Xarray engine string
#             # (Adjust these keys based on what format_from_magic returns)
#             engine_mapping = {
#                 "NETCDF3_CLASSIC": "scipy",
#                 "NETCDF3_64BIT": "scipy",
#                 "NETCDF4": "h5netcdf",
#             }

#             # Explicitly route to the correct engine based on magic bytes
#             engine = engine_mapping.get(self.file_format, None)

#             # Pass the cached file_obj and explicitly state the engine
#             self._dataset = xarray.open_dataset(
#                 self._file_obj,
#                 engine=engine,
#                 decode_cf=False,     # Crucial for metadata-only speed
#                 decode_times=False,
#                 decode_coords=False,
#             )

#         return self._dataset

#     def cleanup(self) -> None:
#         """Safely close down the dataset and the underlying fsspec file handle."""
#         if self._dataset is not None:
#             self._dataset.close()
#             self._dataset = None

#         if self._file_obj is not None:
#             self._file_obj.close()
#             self._file_obj = None

#         self._detected_format = None
