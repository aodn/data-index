from unittest.mock import MagicMock, patch

import polars
import pytest

from data_index.extract import extract
from data_index.protocols import ObjectReference


def _batch_df(rows: list[dict]) -> polars.DataFrame:
    return polars.DataFrame(rows)


@pytest.fixture(autouse=True)
def _patch_prefect_runtime():
    with (
        patch("data_index.extract.prefect.get_run_logger", return_value=MagicMock()),
        patch("data_index.extract.prefect.artifacts.create_table_artifact"),
    ):
        yield


def test_extract_accepts_required_columns_with_extras_and_builds_object_refs():
    batch_df = _batch_df(
        [
            {
                "bucket": "bucket-a",
                "key": "path/a.nc",
                "version_id": "v1",
                "size": 10,
                "extra": "x",
            },
            {
                "bucket": "bucket-b",
                "key": "path/b.nc",
                "version_id": "v2",
                "size": 20,
                "extra": "y",
            },
        ]
    )
    fetcher = MagicMock()
    fetcher.fetch.return_value = []

    extract.fn(batch_df=batch_df, fetcher=fetcher)

    fetch_entries = fetcher.fetch.call_args.args[0]
    assert [entry.object_ref for entry in fetch_entries] == [
        ObjectReference(bucket="bucket-a", key="path/a.nc", version_id="v1"),
        ObjectReference(bucket="bucket-b", key="path/b.nc", version_id="v2"),
    ]
    assert [entry.size_bytes for entry in fetch_entries] == [10, 20]


def test_extract_rejects_duplicate_object_version_identity():
    batch_df = _batch_df(
        [
            {
                "bucket": "bucket-a",
                "key": "path/a.nc",
                "version_id": "v1",
                "size": 10,
            },
            {
                "bucket": "bucket-a",
                "key": "path/a.nc",
                "version_id": "v1",
                "size": 20,
            },
        ]
    )

    with pytest.raises(
        ValueError, match=r"Duplicate \(`bucket`, `key`, `version_id`\) values in batch"
    ):
        extract.fn(batch_df=batch_df, fetcher=MagicMock())


@pytest.mark.parametrize("missing_column", ["bucket", "key", "version_id", "size"])
def test_extract_rejects_missing_required_columns(missing_column: str):
    row = {
        "bucket": "bucket-a",
        "key": "path/a.nc",
        "version_id": "v1",
        "size": 10,
    }
    row.pop(missing_column)
    batch_df = _batch_df([row])

    with pytest.raises(ValueError, match="Batch schema missing required columns"):
        extract.fn(batch_df=batch_df, fetcher=MagicMock())
