from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import prefect
import pytest

from data_index.protocols import ObjectReference
from data_index.runners import index


def _object_reference(key: str) -> ObjectReference:
    return ObjectReference(
        bucket="test",
        key=key,
        version_id="0",
        size=32,
        xarray_handle=None,
    )


def _object_reference_batch() -> list[ObjectReference]:

    return [
        _object_reference(key="IMOS/ANMN/a.nc"),
        _object_reference(key="IMOS/ANMN/b.nc"),
        _object_reference(key="IMOS/ANMN/c.nc"),
    ]


def test_index_batch_raises_when_subflow_state_is_none():
    """Ensure index_batch raises RuntimeError when deployment returns no final state."""
    flow_run = SimpleNamespace(state=None)

    with patch(
        "data_index.runners.index.prefect.deployments.run_deployment",
        return_value=flow_run,
    ):
        with pytest.raises(RuntimeError, match="unknown state"):
            index.index_batch.fn(
                i=0,
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
                object_reference_batch=_object_reference_batch(),
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
            )


def test_index_batch_propagates_failed_subflow_exception():
    """Ensure index_batch propagates the underlying Prefect FailedRun exception."""
    flow_run = SimpleNamespace(state=prefect.states.Failed(message="subflow failed"))

    with patch(
        "data_index.runners.index.prefect.deployments.run_deployment",
        return_value=flow_run,
    ):
        with pytest.raises(prefect.exceptions.FailedRun, match="subflow failed"):
            index.index_batch.fn(
                i=1,
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
                object_reference_batch=_object_reference_batch(),
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
            )


def test_run_index_work_pool_raises_if_any_batch_future_fails():
    """Ensure run_index_work_pool fails overall when any submitted batch future fails."""
    logger = MagicMock()
    inventory_source = MagicMock()
    inventory_source.inventory.return_value = ["a", "b"]

    partitioner = MagicMock()
    partitioner.partition.return_value = [
        _object_reference_batch(),
        _object_reference_batch(),
    ]

    ok_future = SimpleNamespace(state=prefect.states.Completed())
    failed_future = SimpleNamespace(state=prefect.states.Failed(message="batch failed"))

    with (
        patch(
            "data_index.runners.index.prefect.get_run_logger",
            return_value=logger,
        ),
        patch(
            "data_index.runners.index.index_batch.submit",
            side_effect=[ok_future, failed_future],
        ),
        patch(
            "data_index.runners.index.prefect.futures.as_completed",
            return_value=[ok_future, failed_future],
        ),
    ):
        with pytest.raises(RuntimeError, match=r"1 batch\(es\) failed"):
            index.index.fn(
                inventory_source=inventory_source,
                partitioner=partitioner,
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
            )
