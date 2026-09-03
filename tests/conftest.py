from unittest.mock import MagicMock, patch

import pytest
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    """
    Ensures Prefect uses a temporary local SQLite database for testing,
    preventing it from communicating with your real Prefect server/cloud.
    Runs once per test session.
    """
    with prefect_test_harness():
        yield


@pytest.fixture(autouse=True)
def mock_run_logger():
    """
    Globally mocks Prefect's get_run_logger for every test.
    Prevents logger initialization errors and silences terminal spam.
    Runs fresh for every individual test.
    """
    with patch("prefect.get_run_logger") as mock_get_logger:
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        yield mock_logger_instance
