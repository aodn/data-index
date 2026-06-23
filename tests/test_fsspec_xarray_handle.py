from unittest.mock import MagicMock, patch

from data_index.xarray_handle.fsspec_xarray_handle import FSSpecXarrayHandle


def _fsspec_handle(s3_uri: str = "s3://bucket/path/to/file.nc") -> FSSpecXarrayHandle:
    return FSSpecXarrayHandle(
        s3_uri=s3_uri,
    )


def test_s3_uri_returns_correct_string():
    fsspec_handle = _fsspec_handle("s3://my-bucket/data/file.nc")

    assert fsspec_handle.s3_uri == "s3://my-bucket/data/file.nc"


def _mock_fsspec_open(magic: bytes):
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = magic
    mock_fs = MagicMock()
    mock_fs.open.return_value = mock_file
    return mock_fs


def test_file_format_detects_netcdf4():
    fsspec_handle = _fsspec_handle()
    with patch(
        "data_index.xarray_handle.fsspec_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"\x89HDF\r\n\x1a\n"),
    ):
        assert fsspec_handle.file_format == "NETCDF4"


def test_file_format_detects_netcdf3_classic():
    handle = _fsspec_handle()
    with patch(
        "data_index.xarray_handle.fsspec_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"CDF\x01\x00\x00\x00\x00"),
    ):
        assert handle.file_format == "NETCDF3_CLASSIC"


def test_file_format_returns_none_for_unknown():
    handle = _fsspec_handle()
    with patch(
        "data_index.xarray_handle.fsspec_xarray_handle.fsspec.filesystem",
        return_value=_mock_fsspec_open(b"\x00\x01\x02\x03\x04\x05\x06\x07"),
    ):
        assert handle.file_format is None


def test_ds_returns_singleton_dataset():
    fsspec_handle = _fsspec_handle()
    mock_fs = MagicMock()
    mock_file = MagicMock()
    mock_fs.open.return_value = mock_file
    dataset = MagicMock()

    with (
        patch(
            "data_index.xarray_handle.fsspec_xarray_handle.fsspec.filesystem",
            return_value=mock_fs,
        ) as filesystem,
        patch(
            "data_index.xarray_handle.fsspec_xarray_handle.xarray.open_dataset",
            return_value=dataset,
        ) as open_dataset,
    ):
        first = fsspec_handle.ds
        second = fsspec_handle.ds

        # Assertions
        assert first is dataset
        assert second is dataset
        filesystem.assert_called_once_with("s3", anon=True)
        mock_fs.open.assert_called_once_with(
            path=fsspec_handle.s3_uri,
            cache_type=fsspec_handle.cache_type,
            block_size=fsspec_handle.block_size,
        )

        # Reconciled to match the implementation's exact arguments
        open_dataset.assert_called_once_with(
            filename_or_obj=mock_file,
            decode_cf=False,
        )
