from unittest.mock import MagicMock, patch

from data_index.protocols import ExtractionResult
from data_index.transform import transform


class _Handle:
    def __init__(self, s3_uri: str):
        self.s3_uri = s3_uri


def test_transform_runs_sequentially_and_preserves_order():
    handles = [_Handle("s3://bucket/a.nc"), _Handle("s3://bucket/b.nc")]
    extractor = MagicMock()
    metadata_factory = MagicMock()
    expected_results = [
        ExtractionResult(
            s3_uri=handle.s3_uri,
            structured_metadata=None,
            unstructured_metadata=None,
            status="failed",
            error="boom",
        )
        for handle in handles
    ]
    logger = MagicMock()

    with (
        patch(
            "data_index.transform.prefect.get_run_logger",
            return_value=logger,
        ),
        patch(
            "data_index.transform._transform_single",
            side_effect=expected_results,
        ) as single_transform,
    ):
        actual = transform.fn(
            xarray_handles=handles,
            extractor=extractor,
            metadata_factory=metadata_factory,
            max_workers=8,
        )

    transformed_handles = [
        call.kwargs["xarray_handle"] for call in single_transform.call_args_list
    ]
    assert transformed_handles == handles
    assert single_transform.call_count == len(handles)
    assert actual == expected_results
