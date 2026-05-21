from data_index.unstructured_metadata.disk_cache_unstructured_metadata import DiskCachedUnstructuredMetadata


def test_load_returns_stored_data():
    data = {"title": "Test", "count": 42}
    handle = DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/load-test.nc", data=data)

    assert handle.load() == data


def test_two_uris_are_independent():
    DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/uri-a.nc", data={"key": "a"})
    DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/uri-b.nc", data={"key": "b"})

    handle_a = DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/uri-a.nc", data={"key": "a"})
    handle_b = DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/uri-b.nc", data={"key": "b"})

    assert handle_a.load()["key"] == "a"
    assert handle_b.load()["key"] == "b"


def test_overwrite_with_same_uri_returns_latest_data():
    DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/overwrite.nc", data={"version": 1})
    handle = DiskCachedUnstructuredMetadata(s3_uri="s3://bucket/overwrite.nc", data={"version": 2})

    assert handle.load()["version"] == 2
