import pathlib

import pytest

import data_index.file_fetcher
import data_index.protocols
import data_index.xarray_handle


@pytest.fixture
def make_object_reference():
    def _make(
        *,
        bucket: str,
        key: str,
        version_id: str = "__LOCAL__",
        size: int = 32,
    ) -> data_index.protocols.ObjectReference:
        return data_index.protocols.ObjectReference(
            bucket=bucket,
            key=key,
            version_id=version_id,
            size=size,
        )

    return _make


@pytest.fixture
def local_fetcher() -> data_index.file_fetcher.LocalFetcher:
    return data_index.file_fetcher.LocalFetcher(bucket="local-dev-volume")


def test_fetch_returns_staged_objects_for_valid_local_paths(
    tmp_path: pathlib.Path, make_object_reference, local_fetcher
):
    file_path = tmp_path / "sample.nc"
    file_path.write_bytes(b"dummy")

    object_reference = make_object_reference(
        bucket="local-dev-volume",
        key=str(file_path),
    )

    staged_objects, dead_letters = local_fetcher.fetch([object_reference])

    assert len(staged_objects) == 1
    assert dead_letters == []
    assert isinstance(
        staged_objects[0].xarray_handle, data_index.xarray_handle.DiskXarrayHandle
    )
    assert staged_objects[0].xarray_handle.path == file_path
    assert staged_objects[0].xarray_handle.delete_on_cleanup is False


def test_fetch_dead_letters_invalid_entries_and_continues(
    tmp_path: pathlib.Path, make_object_reference, local_fetcher
):
    valid_path = tmp_path / "valid.nc"
    valid_path.write_bytes(b"ok")
    missing_path = tmp_path / "missing.nc"

    object_references = [
        make_object_reference(
            bucket="local-dev-volume",
            key=str(valid_path),
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket="wrong-bucket",
            key=str(valid_path),
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket="local-dev-volume",
            key=str(valid_path),
            version_id="v1",
        ),
        make_object_reference(
            bucket="local-dev-volume",
            key="relative/path.nc",
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket="local-dev-volume",
            key=str(missing_path),
            version_id="__LOCAL__",
        ),
    ]

    staged_objects, dead_letters = local_fetcher.fetch(object_references)

    assert len(staged_objects) == 1
    assert len(dead_letters) == 4

    errors = [dead_letter.error for dead_letter in dead_letters if dead_letter.error]
    assert any("Bucket mismatch" in error for error in errors)
    assert any("Invalid local version_id" in error for error in errors)
    assert any("absolute file path" in error for error in errors)
    assert any("Local file not found" in error for error in errors)
