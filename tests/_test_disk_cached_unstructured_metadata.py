from data_index.protocols import ObjectReference
from data_index.unstructured_metadata.disk_cache_unstructured_metadata import (
    DiskCachedUnstructuredMetadata,
)


def _object_ref(uri: str, version_id: str = "v1") -> ObjectReference:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return ObjectReference(bucket=bucket, key=key, version_id=version_id)


def test_load_returns_stored_data():
    data = {"title": "Test", "count": 42}
    handle = DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/load-test.nc"), data=data
    )

    assert handle.load() == data


def test_two_uris_are_independent():
    DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/uri-a.nc"), data={"key": "a"}
    )
    DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/uri-b.nc"), data={"key": "b"}
    )

    handle_a = DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/uri-a.nc"), data={"key": "a"}
    )
    handle_b = DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/uri-b.nc"), data={"key": "b"}
    )

    assert handle_a.load()["key"] == "a"
    assert handle_b.load()["key"] == "b"


def test_overwrite_with_same_uri_returns_latest_data():
    DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/overwrite.nc"), data={"version": 1}
    )
    handle = DiskCachedUnstructuredMetadata(
        object_ref=_object_ref("s3://bucket/overwrite.nc"), data={"version": 2}
    )

    assert handle.load()["version"] == 2
