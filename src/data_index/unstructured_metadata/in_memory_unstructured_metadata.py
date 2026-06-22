from data_index.protocols import ObjectReference


class InMemoryUnstructuredMetadata:
    """UnstructuredMetadata held in memory. Used by extractors; swapped out by _transform_single."""

    def __init__(self, object_ref: ObjectReference, data: dict) -> None:
        self._data = data

    def load(self) -> dict:
        return self._data
