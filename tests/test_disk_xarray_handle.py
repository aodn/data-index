import pathlib

from data_index.xarray_handle.disk_xarray_handle import DiskXarrayHandle


def test_cleanup_deletes_the_file(tmp_path):
    f = tmp_path / "file.nc"
    f.touch()
    handle = DiskXarrayHandle(path=f, s3_uri="s3://bucket/file.nc")

    handle.cleanup()

    assert not f.exists()


def test_cleanup_is_noop_when_file_does_not_exist(tmp_path):
    path = tmp_path / "nonexistent.nc"
    handle = DiskXarrayHandle(path=path, s3_uri="s3://bucket/file.nc")

    handle.cleanup()  # Should not raise
