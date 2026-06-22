from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars
import prefect
import pytest

from data_index.work_pool_native import index as work_pool_index


def _batch_df(uri: str) -> polars.DataFrame:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return polars.DataFrame(
        {"bucket": [bucket], "key": [key], "version_id": ["v1"], "size": [1]}
    )


def test_index_batch_raises_when_subflow_state_is_none():
    """Ensure index_batch raises RuntimeError when deployment returns no final state."""
    flow_run = SimpleNamespace(state=None)

    with patch(
        "data_index.work_pool_native.index.prefect.deployments.run_deployment",
        return_value=flow_run,
    ):
        with pytest.raises(RuntimeError, match="unknown state"):
            work_pool_index.index_batch.fn(
                i=0,
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
                batch_df=_batch_df("s3://bucket/a.nc"),
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
                transform_max_workers=1,
            )


def test_index_batch_propagates_failed_subflow_exception():
    """Ensure index_batch propagates the underlying Prefect FailedRun exception."""
    flow_run = SimpleNamespace(state=prefect.states.Failed(message="subflow failed"))

    with patch(
        "data_index.work_pool_native.index.prefect.deployments.run_deployment",
        return_value=flow_run,
    ):
        with pytest.raises(prefect.exceptions.FailedRun, match="subflow failed"):
            work_pool_index.index_batch.fn(
                i=1,
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
                batch_df=_batch_df("s3://bucket/b.nc"),
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
                transform_max_workers=1,
            )


def test_run_index_work_pool_raises_if_any_batch_future_fails():
    """Ensure run_index_work_pool fails overall when any submitted batch future fails."""
    logger = MagicMock()
    inventory_source = MagicMock()
    inventory_source.inventory.return_value = ["a", "b"]

    partitioner = MagicMock()
    partitioner.partition.return_value = [
        _batch_df("s3://bucket/a.nc"),
        _batch_df("s3://bucket/b.nc"),
    ]

    ok_future = SimpleNamespace(state=prefect.states.Completed())
    failed_future = SimpleNamespace(state=prefect.states.Failed(message="batch failed"))

    with (
        patch(
            "data_index.work_pool_native.index.prefect.get_run_logger",
            return_value=logger,
        ),
        patch(
            "data_index.work_pool_native.index.index_batch.submit",
            side_effect=[ok_future, failed_future],
        ),
        patch(
            "data_index.work_pool_native.index.prefect.futures.as_completed",
            return_value=[ok_future, failed_future],
        ),
    ):
        with pytest.raises(RuntimeError, match=r"1 batch\(es\) failed"):
            work_pool_index.run_index_work_pool.fn(
                inventory_source=inventory_source,
                partitioner=partitioner,
                fetcher=MagicMock(),
                extractor=MagicMock(),
                structured_sink=MagicMock(),
                unstructured_sink=MagicMock(),
                index_batch_flow_name="index-batch",
                index_batch_deployment_name="index-batch",
                transform_max_workers=1,
            )
