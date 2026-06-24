from unittest.mock import MagicMock, patch

import pytest
from prefect.states import Completed, Failed
from prefect.testing.utilities import prefect_test_harness

from data_index.runners.index import (
    index,
    index_batch,
    index_pipeline,
)


@pytest.fixture(scope="session", autouse=True)
def prefect_test_env():
    """Provides an isolated, in-memory Prefect server ONCE for the entire test suite execution."""
    with prefect_test_harness():
        yield


@pytest.fixture(scope="function")
def mock_batch_dependencies():
    """Generates standard mock objects for the protocol dependencies required by the task."""
    return {
        "fetcher": MagicMock(),
        "extractor": MagicMock(),
        "structured_sink": MagicMock(),
        "unstructured_sink": MagicMock(),
        "object_reference_batch": [
            MagicMock(),
            MagicMock(),
        ],
    }


@pytest.fixture(scope="function")
def mock_index_dependencies():
    """Generates standard mock objects for the protocol dependencies required by the task."""

    return {
        "inventory_source": MagicMock(),
        "partitioner": MagicMock(),
        "fetcher": MagicMock(),
        "extractor": MagicMock(),
        "structured_sink": MagicMock(),
        "unstructured_sink": MagicMock(),
        "index_batch_flow_name": "flow_name",
        "index_batch_deployment_name": "deployment_name",
        "task_runner_config": MagicMock(),
    }


@patch("prefect.deployments.run_deployment")
def test_index_batch_success(mock_run_deployment, mock_batch_dependencies):
    """Verifies index_batch runs successfully when the downstream deployment completes."""
    # Setup mock flow run with a valid completed state
    mock_flow_run = MagicMock()
    mock_flow_run.state = Completed()
    mock_run_deployment.return_value = mock_flow_run

    # Execute the task
    # Note: Calling .fn() bypasses the Prefect scheduler to test the pure Python logic directly,
    # but you can also use index_batch(...) within the harness context.
    index_batch.fn(
        i=1,
        index_batch_flow_name="test-flow",
        index_batch_deployment_name="test-deployment",
        **mock_batch_dependencies,
    )

    # Assert run_deployment was invoked with the exact structural parameters expected
    mock_run_deployment.assert_called_once_with(
        name="test-flow/test-deployment",
        flow_run_name="process-batch-1",
        parameters={
            "object_reference_batch": mock_batch_dependencies["object_reference_batch"],
            "fetcher": mock_batch_dependencies["fetcher"],
            "extractor": mock_batch_dependencies["extractor"],
            "structured_sink": mock_batch_dependencies["structured_sink"],
            "unstructured_sink": mock_batch_dependencies["unstructured_sink"],
        },
    )


@patch("prefect.deployments.run_deployment")
def test_index_batch_unknown_state_exception(
    mock_run_deployment, mock_batch_dependencies
):
    """Verifies index_batch raises a RuntimeError if the flow run yields an unknown state (None)."""
    # Setup mock flow run with state as None
    mock_flow_run = MagicMock()
    mock_flow_run.state = None
    mock_run_deployment.return_value = mock_flow_run

    # Assert that the code explicitly catches the missing state and throws a RuntimeError
    with pytest.raises(RuntimeError, match="finalised with unknown state"):
        index_batch.fn(
            i=42,
            index_batch_flow_name="test-flow",
            index_batch_deployment_name="test-deployment",
            **mock_batch_dependencies,
        )


@patch("prefect.deployments.run_deployment")
@patch("prefect.states.raise_state_exception")
def test_index_batch_bubbles_up_failure(
    mock_raise_state_exception, mock_run_deployment, mock_batch_dependencies
):
    """Verifies index_batch bubbles up orchestration failures utilizing Prefect's state handlers."""
    # Setup mock flow run to return a failed state
    mock_flow_run = MagicMock()
    failed_state = Failed(message="Deployment crashed")
    mock_flow_run.state = failed_state
    mock_run_deployment.return_value = mock_flow_run

    # Configure the state handler mock to raise an error when given a failed state
    mock_raise_state_exception.side_effect = ValueError("State processing exception")

    # Assert that the task fails by letting the exception bubble out
    with pytest.raises(ValueError, match="State processing exception"):
        index_batch.fn(
            i=2,
            index_batch_flow_name="test-flow",
            index_batch_deployment_name="test-deployment",
            **mock_batch_dependencies,
        )

    # Verify the failure state was handed off to Prefect's exception handler properly
    mock_raise_state_exception.assert_called_once_with(failed_state)


@pytest.fixture
def mock_pipeline_dependencies():
    """Generates standard mock dependencies matching the protocol definitions."""
    inventory_source = MagicMock()
    # Mocking inventory data containing 3 sample object references
    inventory_source.inventory.return_value = ["ref1", "ref2", "ref3"]

    partitioner = MagicMock()
    # Mock partitioner to split into 2 batches
    partitioner.partition.return_value = [["ref1", "ref2"], ["ref3"]]

    return {
        "inventory_source": inventory_source,
        "partitioner": partitioner,
        "fetcher": MagicMock(),
        "extractor": MagicMock(),
        "structured_sink": MagicMock(),
        "unstructured_sink": MagicMock(),
    }


@patch("prefect.get_run_logger")
@patch("prefect.futures.as_completed")
@patch.object(index_batch, "submit")
def test_index_pipeline_success(
    mock_submit, mock_as_completed, mock_get_logger, mock_pipeline_dependencies
):
    """Verifies that the entire pipeline runs smoothly, provisions sinks, and partitions data."""
    # Setup mock futures representing successful batch runs
    mock_future_1 = MagicMock(state=Completed())
    mock_future_2 = MagicMock(state=Completed())
    mock_get_logger.return_value = MagicMock()

    mock_submit.side_effect = [mock_future_1, mock_future_2]
    mock_as_completed.return_value = [mock_future_1, mock_future_2]

    # Run the flow orchestration
    index_pipeline.fn(**mock_pipeline_dependencies)

    # Verify sinks were provisioned correctly
    mock_pipeline_dependencies["structured_sink"].provision.assert_called_once()
    mock_pipeline_dependencies["unstructured_sink"].provision.assert_called_once()

    # Verify partitioner was handed the correct inventory
    mock_pipeline_dependencies["partitioner"].partition.assert_called_once_with(
        inventory=["ref1", "ref2", "ref3"]
    )

    # Verify task submission happened exactly twice (one per chunk)
    assert mock_submit.call_count == 2


@patch("prefect.futures.as_completed")
@patch.object(index_batch, "submit")
def test_index_pipeline_aggregates_failures(
    mock_submit, mock_as_completed, mock_pipeline_dependencies
):
    """Verifies index_pipeline gathers all errors and throws a single combined failure exception."""
    mock_future_success = MagicMock(state=Completed())
    mock_future_failed = MagicMock(
        state=Failed(message="Batch 2 processing broke down")
    )

    mock_submit.side_effect = [mock_future_success, mock_future_failed]
    mock_as_completed.return_value = [mock_future_success, mock_future_failed]

    with pytest.raises(RuntimeError):
        index_pipeline.fn(**mock_pipeline_dependencies)


@patch("data_index.runners.index.index_pipeline")
def test_index_applies_task_runner_options(
    mock_index_pipeline, mock_index_dependencies
):
    """Verifies that the task runner config is generated and applied via .with_options()"""
    # Setup the mock runner instance that the config block will return
    mock_runner_instance = MagicMock()
    mock_index_dependencies[
        "task_runner_config"
    ].create.return_value = mock_runner_instance

    # Set up the fluent interface mocking for index_pipeline.with_options()(parameters)
    mock_configured_pipeline = MagicMock()
    mock_index_pipeline.with_options.return_value = mock_configured_pipeline

    # Execute the entrypoint flow
    index.fn(
        inventory_source=mock_index_dependencies["inventory_source"],
        partitioner=mock_index_dependencies["partitioner"],
        fetcher=mock_index_dependencies["fetcher"],
        extractor=mock_index_dependencies["extractor"],
        structured_sink=mock_index_dependencies["structured_sink"],
        unstructured_sink=mock_index_dependencies["unstructured_sink"],
        task_runner_config=mock_index_dependencies["task_runner_config"],
        index_batch_flow_name="custom-flow",
        index_batch_deployment_name="custom-deployment",
    )

    # Assert that the runner instance was created from the configuration block
    mock_index_dependencies["task_runner_config"].create.assert_called_once()

    # Assert that .with_options was called to pass down that exact task runner instance
    mock_index_pipeline.with_options.assert_called_once_with(
        task_runner=mock_runner_instance,
    )

    # Assert that the resulting configured pipeline was executed with the original parameters
    mock_configured_pipeline.assert_called_once_with(
        inventory_source=mock_index_dependencies["inventory_source"],
        partitioner=mock_index_dependencies["partitioner"],
        fetcher=mock_index_dependencies["fetcher"],
        extractor=mock_index_dependencies["extractor"],
        structured_sink=mock_index_dependencies["structured_sink"],
        unstructured_sink=mock_index_dependencies["unstructured_sink"],
        index_batch_flow_name="custom-flow",
        index_batch_deployment_name="custom-deployment",
    )
