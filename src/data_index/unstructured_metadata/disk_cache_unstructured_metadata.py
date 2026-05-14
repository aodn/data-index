import pathlib
import typing
import diskcache

class DiskCachedUnstructuredMetadata:
    """UnstructuredMetadata written immediately to diskcache on construction.

    The raw dict is not retained in memory. Call load() to read it back from disk.
    Configure the cache location and size by setting CACHE_PATH / CACHE_SIZE_LIMIT
    on the class before use.
    """

    CACHE_PATH: typing.ClassVar[pathlib.Path] = pathlib.Path(".transform")
    CACHE_SIZE_LIMIT: typing.ClassVar[int] = int(10e9)  # 10 GB

    def __init__(self, s3_uri: str, data: dict, cache_path: pathlib.Path | None = None) -> None:
        resolved_path = cache_path if cache_path is not None else self.CACHE_PATH
        with diskcache.Cache(str(resolved_path), size_limit=self.CACHE_SIZE_LIMIT) as cache:
            cache[s3_uri] = data
        self._s3_uri = s3_uri
        self._cache_path = resolved_path

    def load(self) -> dict:
        with diskcache.Cache(str(self._cache_path), size_limit=self.CACHE_SIZE_LIMIT) as cache:
            return cache[self._s3_uri]