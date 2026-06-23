from data_index.file_fetcher.fsspec_fetcher import FSSpecFetcher
from data_index.protocols import ObjectReference


def _object_reference(
    bucket: str = "test",
    key: str = "IMOS/file.nc",
    version_id: str = "0",
    size: int = 32,
) -> ObjectReference:
    return ObjectReference(
        bucket=bucket,
        key=key,
        version_id=version_id,
        size=size,
        xarray_handle=None,
    )


def test_returns_one_handle_per_uri():
    fetcher = FSSpecFetcher()
    object_references = [_object_reference(key=key) for key in ["a.nc", "b.nc", "c.nc"]]

    object_references = fetcher.fetch(object_references)

    assert len(object_references) == 3


def test_returns_empty_list_for_empty_input():
    fetcher = FSSpecFetcher()

    handles = fetcher.fetch([])

    assert handles == []
