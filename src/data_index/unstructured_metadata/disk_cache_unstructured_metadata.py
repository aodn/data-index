import pathlib
import typing
import tempfile
import diskcache
import atexit

class DiskCachedUnstructuredMetadata:
    """UnstructuredMetadata written to a strictly temporary diskcache.
    
    The cache directory is automatically deleted when the python process exits.
    """

    CACHE_SIZE_LIMIT: typing.ClassVar[int] = int(10e9)
    _temp_dir_manager = tempfile.TemporaryDirectory(prefix="metadata_cache_")
    CACHE_PATH: pathlib.Path = pathlib.Path(_temp_dir_manager.name)

    def __init__(self, s3_uri: str, data: dict) -> None:
        self._s3_uri = s3_uri
        with diskcache.Cache(str(self.CACHE_PATH), size_limit=self.CACHE_SIZE_LIMIT) as cache:
            cache[s3_uri] = data

    def load(self) -> dict:
        with diskcache.Cache(str(self.CACHE_PATH), size_limit=self.CACHE_SIZE_LIMIT) as cache:
            return cache[self._s3_uri]

    @classmethod
    def cleanup(cls):
        """Explicitly trigger cleanup if needed before process end."""
        cls._temp_dir_manager.cleanup()

atexit.register(DiskCachedUnstructuredMetadata.cleanup)