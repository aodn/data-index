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
