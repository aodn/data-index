from unittest.mock import patch

import pytest

from data_index.protocols import (
    DeadLetter,
    ObjectReference,
)


@pytest.fixture(scope="session")
def object_reference():
    """Provides a valid ObjectReference instance for metadata tests."""
    return ObjectReference(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        size=0,
        xarray_handle=None,
    )


@patch("prefect.runtime.flow_run.get_id")
@patch("prefect.runtime.flow_run.get_parent_flow_run_id")
def test_prefect_runtime_defaults_success(mock_get_parent, mock_get_id):
    """Verify context-aware IDs are resolved when running inside an active Prefect flow."""
    mock_get_parent.return_value = "flow-parent-abc-123"
    mock_get_id.return_value = "flow-batch-xyz-789"

    dl = DeadLetter(
        error="Pipeline failure",
        bucket="my-bucket",
        key="path/file.json",
        version_id=None,
        size=2048,
    )

    assert dl.index_flow_id == "flow-parent-abc-123"
    assert dl.batch_flow_id == "flow-batch-xyz-789"
    mock_get_parent.assert_called_once()
    mock_get_id.assert_called_once()


@patch("prefect.runtime.flow_run.get_id")
@patch("prefect.runtime.flow_run.get_parent_flow_run_id")
def test_prefect_runtime_defaults_missing_context(mock_get_parent, mock_get_id):
    """Verify fields default gracefully to None if executed outside of a Prefect runtime environment."""
    mock_get_parent.return_value = None
    mock_get_id.return_value = None
    dl = DeadLetter(
        error="Local failure",
        bucket="my-bucket",
        key="path/file.json",
        version_id=None,
        size=2048,
    )

    assert dl.index_flow_id is None
    assert dl.batch_flow_id is None


def test_prefect_runtime_explicit_overrides():
    """Verify that passing explicit flow IDs overrides the Prefect runtime factories entirely."""
    dl = DeadLetter(
        error="Manual run",
        bucket="b",
        key="k",
        version_id=None,
        size=10,
        index_flow_id="explicit-parent",
        batch_flow_id="explicit-batch",
    )

    assert dl.index_flow_id == "explicit-parent"
    assert dl.batch_flow_id == "explicit-batch"


def test_from_object_reference_mapping(object_reference):
    """Verify property mapping from an ObjectReference compliant object."""

    error = "parquet format corruption detected"

    dead_letter = DeadLetter.from_object_reference(
        object_reference=object_reference,
        error=error,
    )

    assert dead_letter.bucket == object_reference.bucket
    assert dead_letter.key == object_reference.key
    assert dead_letter.version_id == object_reference.version_id
    assert dead_letter.size == object_reference.size
    assert dead_letter.error == error


def test_from_object_reference_with_null_error(object_reference):
    """Verify the factory method accepts a None value for the error parameter."""

    error = None

    dead_letter = DeadLetter.from_object_reference(
        object_reference=object_reference,
        error=error,
    )

    assert dead_letter.bucket == object_reference.bucket
    assert dead_letter.key == object_reference.key
    assert dead_letter.version_id == object_reference.version_id
    assert dead_letter.size == object_reference.size
    assert dead_letter.error is error
