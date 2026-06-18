from unittest.mock import MagicMock, patch

from data_index.protocols import ExtractionResult
from data_index.transform import transform


class _Executor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.mapped = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def map(self, fn, iterable):
        items = list(iterable)
        self.mapped = [(fn, item) for item in items]
        return [fn(item) for item in items]


class _Handle:
    def __init__(self, s3_uri: str):
        self.s3_uri = s3_uri


def test_transform_uses_thread_pool_and_honors_max_workers():
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
    executor = _Executor()

    with (
        patch(
            "data_index.transform.prefect.get_run_logger",
            return_value=logger,
        ),
        patch(
            "data_index.transform.prefect.artifacts.create_progress_artifact"
        ) as create_progress,
        patch(
            "data_index.transform.prefect.artifacts.update_progress_artifact"
        ) as update_progress,
        patch("data_index.transform.prefect.artifacts.create_table_artifact"),
        patch(
            "data_index.transform._transform_single",
            side_effect=expected_results,
        ) as single_transform,
        patch(
            "data_index.transform.concurrent.futures.ThreadPoolExecutor",
            return_value=executor,
        ) as pool_ctor,
    ):
        actual = transform.fn(
            xarray_handles=handles,
            extractor=extractor,
            metadata_factory=metadata_factory,
            max_workers=8,
        )

    assert pool_ctor.call_args.kwargs["max_workers"] == 8
    assert single_transform.call_count == len(handles)
    assert len(executor.mapped) == len(handles)
    create_progress.assert_not_called()
    update_progress.assert_not_called()
    assert actual == expected_results
