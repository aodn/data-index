import pytest

import data_index.file_fetcher
import data_index.protocols


@pytest.fixture
def make_object_reference():
    def _make(
        bucket: str = "test",
        key: str = "IMOS/file.nc",
        version_id: str = "0",
        size: int = 32,
    ) -> data_index.protocols.ObjectReference:
        return data_index.protocols.ObjectReference(
            bucket=bucket,
            key=key,
            version_id=version_id,
            size=size,
        )

    return _make


@pytest.fixture(scope="session")
def fsspec_fetcher() -> data_index.file_fetcher.FSSpecFetcher:
    return data_index.file_fetcher.FSSpecFetcher()


@pytest.mark.parametrize(
    "keys, expected_count",
    [
        (["a.nc", "b.nc", "c.nc"], 3),
        (["single_file.nc"], 1),
        (["x.nc", "y.nc"], 2),
    ],
)
def test_returns_one_handle_per_uri(
    make_object_reference, keys, expected_count, fsspec_fetcher
):

    # Use the factory fixture to spin up your objects dynamically
    object_references = [make_object_reference(key=k) for k in keys]

    staged_objects, dead_letters = fsspec_fetcher.fetch(object_references)
    assert len(staged_objects) == expected_count
    assert dead_letters == []


def test_returns_empty_list_for_empty_input(fsspec_fetcher):
    staged_objects, dead_letters = fsspec_fetcher.fetch([])
    assert staged_objects == []
    assert dead_letters == []
