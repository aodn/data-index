import atexit
import pathlib
import tempfile
import typing

import diskcache

from data_index.protocols import ObjectReference


class DiskCachedUnstructuredMetadata:
    """UnstructuredMetadata written to a strictly temporary diskcache.

    The cache directory is automatically deleted when the python process exits.
    """

    CACHE_SIZE_LIMIT: typing.ClassVar[int] = int(10e9)
    _temp_dir_manager = tempfile.TemporaryDirectory(prefix="metadata_cache_")
    CACHE_PATH: pathlib.Path = pathlib.Path(_temp_dir_manager.name)

    def __init__(self, object_ref: ObjectReference, data: dict) -> None:
        self._object_ref = object_ref
        with diskcache.Cache(
            str(self.CACHE_PATH), size_limit=self.CACHE_SIZE_LIMIT
        ) as cache:
            cache[self._cache_key] = data

    @property
    def _cache_key(self) -> tuple[str, str, str]:
        return (
            self._object_ref.bucket,
            self._object_ref.key,
            self._object_ref.version_id,
        )

    def load(self) -> dict:
        with diskcache.Cache(
            str(self.CACHE_PATH), size_limit=self.CACHE_SIZE_LIMIT
        ) as cache:
            return cache[self._cache_key]

    @classmethod
    def cleanup(cls):
        """Explicitly trigger cleanup if needed before process end."""
        cls._temp_dir_manager.cleanup()


atexit.register(DiskCachedUnstructuredMetadata.cleanup)
