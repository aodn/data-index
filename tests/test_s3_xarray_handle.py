from unittest.mock import MagicMock, patch

import cloudpathlib

from data_index.xarray_handle.s3_xarray_handle import S3XarrayHandle


def make_handle(uri: str = "s3://bucket/path/to/file.nc") -> S3XarrayHandle:
    return S3XarrayHandle(path=cloudpathlib.S3Path(uri))


def test_s3_uri_returns_correct_string():
    handle = make_handle("s3://my-bucket/data/file.nc")

    assert handle.s3_uri == "s3://my-bucket/data/file.nc"


def test_cleanup_is_noop():
    handle = make_handle()

    handle.cleanup()  # Should not raise or have side effects


def _mock_fsspec_open(magic: bytes):
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = magic
    mock_fs = MagicMock()
    mock_fs.open.return_value = mock_file
    return mock_fs


def test_file_format_detects_netcdf4():
    handle = make_handle()
    with patch(
        "data_index.xarray_handle.s3_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"\x89HDF\r\n\x1a\n"),
    ):
        assert handle.file_format == "NETCDF4"


def test_file_format_detects_netcdf3_classic():
    handle = make_handle()
    with patch(
        "data_index.xarray_handle.s3_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"CDF\x01\x00\x00\x00\x00"),
    ):
        assert handle.file_format == "NETCDF3_CLASSIC"


def test_file_format_returns_none_for_unknown():
    handle = make_handle()
    with patch(
        "data_index.xarray_handle.s3_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"\x00\x01\x02\x03\x04\x05\x06\x07"),
    ):
        assert handle.file_format is None
