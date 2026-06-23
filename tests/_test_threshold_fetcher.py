import pathlib
from unittest.mock import patch

import pydantic
import pytest

from data_index.file_fetcher.fsspec_fetcher import S3Fetcher
from data_index.file_fetcher.s5cmd_fetcher import S5CMDFetcher
from data_index.file_fetcher.threshold_fetcher import ThresholdFileFetcher
from data_index.protocols import BatchEntry, ObjectReference
from data_index.xarray_handle.disk_xarray_handle import DiskXarrayHandle
from data_index.xarray_handle.fsspec_xarray_handle import S3XarrayHandle

# --- Stubs ---


class _StubDiskFetcher(S5CMDFetcher):
    received: list = pydantic.Field(default_factory=list)

    def fetch(self, entries: list[BatchEntry]):
        self.received = list(entries)
        return [
            DiskXarrayHandle(path=_fake_path(e.uri), object_ref=e.object_ref)
            for e in entries
        ]


class _StubCloudFetcher(S3Fetcher):
    received: list = pydantic.Field(default_factory=list)

    def fetch(self, entries: list[BatchEntry]):
        self.received = list(entries)
        import cloudpathlib

        return [
            S3XarrayHandle(path=cloudpathlib.S3Path(e.uri), object_ref=e.object_ref)
            for e in entries
        ]


def _fake_path(uri: str):
    return pathlib.Path("/tmp") / uri.lstrip("s3://")


def _entry(uri: str, size_bytes: int | None) -> BatchEntry:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return BatchEntry(
        object_ref=ObjectReference(bucket=bucket, key=key, version_id="v1"),
        size_bytes=size_bytes,
    )


# --- Tests ---


@pytest.fixture
def fetcher():
    disk = _StubDiskFetcher()
    cloud = _StubCloudFetcher()
    with patch(
        "data_index.file_fetcher.threshold_fetcher.prefect.artifacts.create_table_artifact"
    ):
        yield (
            ThresholdFileFetcher(
                size_threshold_bytes=100, disk_fetcher=disk, cloud_fetcher=cloud
            ),
            disk,
            cloud,
        )


def test_fetch_returns_empty_for_empty_input(fetcher):
    threshold_fetcher, disk, cloud = fetcher

    result = threshold_fetcher.fetch([])

    assert result == []


def test_large_file_routed_to_cloud_fetcher(fetcher):
    threshold_fetcher, disk, cloud = fetcher
    entry = _entry("s3://bucket/big.nc", size_bytes=200)

    handles = threshold_fetcher.fetch([entry])

    assert len(handles) == 1
    assert isinstance(handles[0], S3XarrayHandle)
    assert cloud.received == [entry]
    assert disk.received == []


def test_small_file_routed_to_disk_fetcher(fetcher):
    threshold_fetcher, disk, cloud = fetcher
    entry = _entry("s3://bucket/small.nc", size_bytes=50)

    handles = threshold_fetcher.fetch([entry])

    assert len(handles) == 1
    assert isinstance(handles[0], DiskXarrayHandle)
    assert disk.received == [entry]
    assert cloud.received == []


def test_mixed_entries_routed_to_correct_fetchers(fetcher):
    threshold_fetcher, disk, cloud = fetcher
    small = _entry("s3://bucket/small.nc", size_bytes=50)
    large = _entry("s3://bucket/big.nc", size_bytes=200)

    handles = threshold_fetcher.fetch([small, large])

    assert len(handles) == 2
    assert any(isinstance(h, DiskXarrayHandle) for h in handles)
    assert any(isinstance(h, S3XarrayHandle) for h in handles)
    assert disk.received == [small]
    assert cloud.received == [large]


def test_none_size_routed_to_cloud_fetcher(fetcher):
    threshold_fetcher, disk, cloud = fetcher
    entry = _entry("s3://bucket/unknown.nc", size_bytes=None)

    handles = threshold_fetcher.fetch([entry])

    assert len(handles) == 1
    assert isinstance(handles[0], S3XarrayHandle)
    assert cloud.received == [entry]
    assert disk.received == []


def test_entry_at_threshold_routed_to_cloud_fetcher(fetcher):
    """size_bytes == threshold is treated as 'large' → cloud."""
    threshold_fetcher, disk, cloud = fetcher
    entry = _entry("s3://bucket/exact.nc", size_bytes=100)

    handles = threshold_fetcher.fetch([entry])

    assert len(handles) == 1
    assert isinstance(handles[0], S3XarrayHandle)
