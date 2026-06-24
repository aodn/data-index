from unittest.mock import MagicMock, patch

from data_index.runners.index_batch import index_batch


# Patch the functions. Note: Patch decorators are applied bottom-up,
# so the order of arguments in the test function matches the order of patches.
@patch("data_index.load")
@patch("data_index.transform")
@patch("data_index.extract")
def test_index_batch_etl_pipeline(
    mock_extract,
    mock_transform,
    mock_load,
):
    # Setup mock return values to track data flow
    # This ensures `transform` receives what `extract` outputs, etc.
    mock_extract.return_value = ["mock_extracted_ref_1", "mock_extracted_ref_2"]
    mock_transform.return_value = ["mock_transformed_result"]

    # Create dummy input arguments
    dummy_batch = []
    dummy_fetcher = MagicMock()
    dummy_extractor = MagicMock()
    dummy_structured_sink = MagicMock()
    dummy_unstructured_sink = MagicMock()

    # Call the flow
    index_batch.fn(
        object_reference_batch=dummy_batch,
        fetcher=dummy_fetcher,
        extractor=dummy_extractor,
        structured_sink=dummy_structured_sink,
        unstructured_sink=dummy_unstructured_sink,
    )

    # Verify `extract` was called with initial inputs
    mock_extract.assert_called_once_with(
        object_references=dummy_batch,
        fetcher=dummy_fetcher,
    )

    # Verify `transform` was called with `extract`'s output
    mock_transform.assert_called_once_with(
        object_references=mock_extract.return_value,
        extractor=dummy_extractor,
    )

    # Verify `load` was called with `transform`'s output
    mock_load.assert_called_once_with(
        extraction_results=mock_transform.return_value,
        structured_sink=dummy_structured_sink,
        unstructured_sink=dummy_unstructured_sink,
    )
