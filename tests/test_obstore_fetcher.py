import pathlib
from unittest.mock import MagicMock, patch

import pytest

from data_index.file_fetcher import ObstoreFetcher


@pytest.fixture
def mock_s3_store():
    """Patches S3Store so we never hit real AWS infrastructure during tests."""
    with patch("obstore.store.S3Store") as mock:
        yield mock


@pytest.fixture
def fetcher(mock_s3_store, tmp_path):
    """Provides a standard ObstoreFetcher instance isolated to a temp directory."""
    return ObstoreFetcher(
        extract_path=tmp_path / ".extract",
        bucket="test-bucket",
        region="us-east-1",
        skip_signature=True,
    )


def test_initialization_and_store_property(mock_s3_store, tmp_path):
    """Verify that Pydantic properly initializes the private S3Store attribute."""
    fetcher_instance = ObstoreFetcher(
        extract_path=tmp_path,
        bucket="custom-bucket",
        region="us-west-2",
        skip_signature=False,
    )

    # Assert the private store was initialized with the correct configuration
    mock_s3_store.assert_called_once_with(
        bucket="custom-bucket", region="us-west-2", skip_signature=False
    )

    # Assert the read-only property exposes the internal private store
    assert fetcher_instance.store == mock_s3_store.return_value


def test_get_stream_without_version(fetcher):
    """Verify get_stream behavior when no version_id is provided."""
    mock_ref = MagicMock()
    mock_ref.key = "data/daily_measurements.nc"
    mock_ref.version_id = None

    # Mock the chained obstore call: store.get().stream()
    mock_stream_target = MagicMock()
    fetcher.store.get.return_value.stream.return_value = mock_stream_target

    result = fetcher.get_stream(mock_ref)

    # Ensure get() was called with empty options
    fetcher.store.get.assert_called_once_with(
        path="data/daily_measurements.nc", options={}
    )
    assert result == mock_stream_target


def test_get_stream_with_version(fetcher):
    """Verify get_stream passes the version option when version_id exists."""
    mock_ref = MagicMock()
    mock_ref.key = "data/daily_measurements.nc"
    mock_ref.version_id = "v-999-alpha"

    fetcher.get_stream(mock_ref)

    # Ensure options dictionary contains the version parameter
    fetcher.store.get.assert_called_once_with(
        path="data/daily_measurements.nc", options={"version": "v-999-alpha"}
    )


def test_stream_to_disk(fetcher, monkeypatch):
    """Verify that chunks are read from the stream and written cleanly to disk."""
    mock_ref = MagicMock()
    mock_ref.path = "subfolder/dataset.nc"

    dummy_chunks = [b"chunk_one_", b"chunk_two"]
    monkeypatch.setattr(
        ObstoreFetcher, "get_stream", lambda self, object_reference: dummy_chunks
    )

    expected_write_path = fetcher.extract_path / "subfolder/dataset.nc"
    resulting_path = fetcher.stream_to_disk(mock_ref)

    assert resulting_path == expected_write_path
    assert expected_write_path.exists()
    assert expected_write_path.read_bytes() == b"chunk_one_chunk_two"


def test_object_reference_to_disk_xarray_handle(fetcher, monkeypatch):
    """Verify interaction between stream_to_disk and DiskXarrayHandle instantiation."""
    mock_ref = MagicMock()
    fake_disk_path = pathlib.Path("/fake/dir/.extract/file.nc")

    monkeypatch.setattr(
        ObstoreFetcher, "stream_to_disk", lambda self, object_reference: fake_disk_path
    )

    with patch("data_index.xarray_handle.DiskXarrayHandle") as mock_handle_cls:
        result = fetcher.object_reference_to_disk_xarray_handle(mock_ref)
        mock_handle_cls.assert_called_once_with(path=fake_disk_path)
        assert result == mock_handle_cls.return_value


def test_fetch_processes_multiple_references(fetcher, monkeypatch):
    """Verify fetch loops over all references and updates them sequentially."""
    mock_ref_1 = MagicMock()
    mock_ref_2 = MagicMock()
    fake_handle = MagicMock()

    monkeypatch.setattr(
        ObstoreFetcher,
        "object_reference_to_disk_xarray_handle",
        lambda self, object_reference: fake_handle,
    )

    mock_ref_1.with_xarray_handle.return_value = "processed_ref_1"
    mock_ref_2.with_xarray_handle.return_value = "processed_ref_2"

    results = fetcher.fetch([mock_ref_1, mock_ref_2])

    mock_ref_1.with_xarray_handle.assert_called_once_with(xarray_handle=fake_handle)
    mock_ref_2.with_xarray_handle.assert_called_once_with(xarray_handle=fake_handle)
    assert results == ["processed_ref_1", "processed_ref_2"]
