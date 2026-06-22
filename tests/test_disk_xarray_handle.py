from unittest.mock import MagicMock, patch

from data_index.protocols import ObjectReference
from data_index.xarray_handle.disk_xarray_handle import DiskXarrayHandle


def _object_ref(uri: str, version_id: str = "v1") -> ObjectReference:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return ObjectReference(bucket=bucket, key=key, version_id=version_id)


def test_cleanup_deletes_the_file(tmp_path):
    f = tmp_path / "file.nc"
    f.touch()
    handle = DiskXarrayHandle(path=f, object_ref=_object_ref("s3://bucket/file.nc"))

    handle.cleanup()

    assert not f.exists()


def test_cleanup_is_noop_when_file_does_not_exist(tmp_path):
    path = tmp_path / "nonexistent.nc"
    handle = DiskXarrayHandle(path=path, object_ref=_object_ref("s3://bucket/file.nc"))

    handle.cleanup()  # Should not raise


def test_file_format_detects_netcdf3_classic(tmp_path):
    f = tmp_path / "nc3.nc"
    f.write_bytes(b"CDF\x01" + b"\x00" * 4)
    handle = DiskXarrayHandle(path=f, object_ref=_object_ref("s3://bucket/nc3.nc"))

    assert handle.file_format == "NETCDF3_CLASSIC"


def test_file_format_detects_netcdf4(tmp_path):
    f = tmp_path / "nc4.nc"
    f.write_bytes(b"\x89HDF\r\n\x1a\n")
    handle = DiskXarrayHandle(path=f, object_ref=_object_ref("s3://bucket/nc4.nc"))

    assert handle.file_format == "NETCDF4"


def test_file_format_returns_none_for_unknown_bytes(tmp_path):
    f = tmp_path / "unknown.bin"
    f.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07")
    handle = DiskXarrayHandle(path=f, object_ref=_object_ref("s3://bucket/unknown.nc"))

    assert handle.file_format is None


def test_ds_returns_singleton_dataset(tmp_path):
    f = tmp_path / "singleton.nc"
    f.touch()
    handle = DiskXarrayHandle(
        path=f, object_ref=_object_ref("s3://bucket/singleton.nc")
    )
    dataset = MagicMock()

    with patch(
        "data_index.xarray_handle.disk_xarray_handle.xarray.open_dataset",
        return_value=dataset,
    ) as open_dataset:
        first = handle.ds
        second = handle.ds

    assert first is dataset
    assert second is dataset
    open_dataset.assert_called_once_with(filename_or_obj=f)
