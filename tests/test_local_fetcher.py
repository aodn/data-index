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
    return data_index.file_fetcher.LocalFetcher()


def test_fetch_returns_staged_objects_for_valid_local_paths(
    tmp_path: pathlib.Path, make_object_reference, local_fetcher
):
    bucket_root = tmp_path / "local-dev-volume"
    bucket_root.mkdir(parents=True, exist_ok=True)
    file_path = bucket_root / "sample.nc"
    file_path.write_bytes(b"dummy")

    object_reference = make_object_reference(
        bucket=str(bucket_root),
        key="sample.nc",
    )

    staged_objects, dead_letters = local_fetcher.fetch([object_reference])

    assert len(staged_objects) == 1
    assert dead_letters == []
    assert isinstance(
        staged_objects[0].xarray_handle, data_index.xarray_handle.DiskXarrayHandle
    )
    assert staged_objects[0].xarray_handle.path == file_path
    assert staged_objects[0].xarray_handle.delete_on_cleanup is False


def test_fetch_stitches_bucket_and_relative_key_to_absolute_path(
    tmp_path: pathlib.Path, make_object_reference
):
    bucket_root = tmp_path / "imos-data"
    file_path = bucket_root / "IMOS" / "Argo" / "sample.nc"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"dummy")

    fetcher = data_index.file_fetcher.LocalFetcher()
    object_reference = make_object_reference(
        bucket=str(bucket_root),
        key="IMOS/Argo/sample.nc",
    )

    staged_objects, dead_letters = fetcher.fetch([object_reference])

    assert len(staged_objects) == 1
    assert dead_letters == []
    assert staged_objects[0].xarray_handle.path == file_path


def test_fetch_dead_letters_invalid_entries_and_continues(
    tmp_path: pathlib.Path, make_object_reference, local_fetcher
):
    bucket_root = tmp_path / "local-dev-volume"
    bucket_root.mkdir(parents=True, exist_ok=True)
    valid_path = bucket_root / "valid.nc"
    valid_path.write_bytes(b"ok")
    outside_path = tmp_path / "outside.nc"
    outside_path.write_bytes(b"outside")

    object_references = [
        make_object_reference(
            bucket=str(bucket_root),
            key="valid.nc",
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket="wrong-bucket",
            key="valid.nc",
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket=str(bucket_root),
            key="valid.nc",
            version_id="v1",
        ),
        make_object_reference(
            bucket=str(bucket_root),
            key=str(valid_path),
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket=str(bucket_root),
            key="missing.nc",
            version_id="__LOCAL__",
        ),
        make_object_reference(
            bucket=str(bucket_root),
            key="../outside.nc",
            version_id="__LOCAL__",
        ),
    ]

    staged_objects, dead_letters = local_fetcher.fetch(object_references)

    assert len(staged_objects) == 1
    assert len(dead_letters) == 5

    errors = [dead_letter.error for dead_letter in dead_letters if dead_letter.error]
    assert any("Local bucket must be an absolute path" in error for error in errors)
    assert any("Invalid local version_id" in error for error in errors)
    assert any("Local key must be a relative path suffix" in error for error in errors)
    assert any("Local file not found" in error for error in errors)
    assert any("Local key escapes bucket path" in error for error in errors)
