from unittest.mock import MagicMock, patch

import polars

from data_index.work_pool_native.index_batch import (
    _ARTIFACT_SAMPLE_LIMIT,
    _summarise_batch_handles,
)


class _DiskHandle:
    def __init__(self, s3_uri: str):
        self.s3_uri = s3_uri


class _CloudHandle:
    def __init__(self, s3_uri: str):
        self.s3_uri = s3_uri


def _batch_df(uris: list[str]) -> polars.DataFrame:
    return polars.DataFrame({"s3_uri": uris, "size": [1] * len(uris)})


def test_summarise_batch_handles_writes_summary_only_for_perfect_coverage():
    logger = MagicMock()
    batch_df = _batch_df(["s3://bucket/a.nc", "s3://bucket/b.nc"])
    handles = [
        _DiskHandle("s3://bucket/a.nc"),
        _CloudHandle("s3://bucket/b.nc"),
        _CloudHandle("s3://bucket/b.nc"),
    ]

    with (
        patch(
            "data_index.work_pool_native.index_batch.prefect.artifacts.create_table_artifact"
        ) as create_table_artifact,
        patch(
            "data_index.work_pool_native.index_batch.prefect.get_run_logger",
            return_value=logger,
        ),
    ):
        _summarise_batch_handles(batch_df=batch_df, handles=handles)

    assert create_table_artifact.call_count == 1
    summary = create_table_artifact.call_args.kwargs
    assert summary["key"] == "extract-handle-summary"
    row = summary["table"][0]
    assert row["expected_uris"] == 2
    assert row["fetched_handles"] == 3
    assert row["unique_fetched_uris"] == 2
    assert row["missing_uris"] == 0
    assert row["extra_uris"] == 0
    assert row["duplicate_handle_uris"] == 1
    logger.warning.assert_not_called()


def test_summarise_batch_handles_caps_mismatch_artifact_samples():
    logger = MagicMock()
    expected = [f"s3://bucket/expected-{i}.nc" for i in range(40)]
    fetched = expected[:5] + [f"s3://bucket/extra-{i}.nc" for i in range(30)]
    batch_df = _batch_df(expected)
    handles = [_CloudHandle(uri) for uri in fetched]

    with (
        patch(
            "data_index.work_pool_native.index_batch.prefect.artifacts.create_table_artifact"
        ) as create_table_artifact,
        patch(
            "data_index.work_pool_native.index_batch.prefect.get_run_logger",
            return_value=logger,
        ),
    ):
        _summarise_batch_handles(batch_df=batch_df, handles=handles)

    artifacts_by_key = {
        call.kwargs["key"]: call.kwargs for call in create_table_artifact.call_args_list
    }

    assert set(artifacts_by_key) == {
        "extract-handle-summary",
        "extract-missing-handle-sample",
        "extract-extra-handle-sample",
    }

    summary = artifacts_by_key["extract-handle-summary"]["table"][0]
    assert summary["expected_uris"] == 40
    assert summary["fetched_handles"] == 35
    assert summary["unique_fetched_uris"] == 35
    assert summary["missing_uris"] == 35
    assert summary["extra_uris"] == 30

    missing_sample = artifacts_by_key["extract-missing-handle-sample"]["table"]
    extra_sample = artifacts_by_key["extract-extra-handle-sample"]["table"]
    assert len(missing_sample) == _ARTIFACT_SAMPLE_LIMIT
    assert len(extra_sample) == _ARTIFACT_SAMPLE_LIMIT
    logger.warning.assert_called_once()
