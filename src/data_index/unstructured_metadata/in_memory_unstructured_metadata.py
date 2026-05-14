class InMemoryUnstructuredMetadata:
    """UnstructuredMetadata held in memory. Used by extractors; swapped out by _transform_single."""

    def __init__(self, s3_uri: str, data: dict) -> None:
        self._data = data

    def load(self) -> dict:
        return self._data