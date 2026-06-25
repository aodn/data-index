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


def _df() -> polars.DataFrame:
    return polars.DataFrame(
        data=[
            {
                "bucket": "bucket-a",
                "key": "path/a.nc",
                "version_id": "v1",
                "size": 10,
            },
            {
                "bucket": "bucket-b",
                "key": "path/b.nc",
                "version_id": "v2",
                "size": 20,
            },
        ]
    )


def _object_references(df: polars.DataFrame = _df()) -> list[ObjectReference]:
    return [
        ObjectReference(
            bucket=bucket,
            key=key,
            version_id=version_id,
            size=size,
        )
        for bucket, key, version_id, size in df.select(
            "bucket", "key", "version_id", "size"
        ).iter_rows()
    ]


def test_extract_rejects_duplicate_object_version_identity():
    object_references = _object_references(
        df=polars.DataFrame(
            data=[
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
    )

    # Updated to match the new dynamic, multi-line error format
    expected_error_regex = (
        r"Validation failed: Duplicate object references detected\.\n"
        r"Duplicates list:\n"
        r"s3://bucket-a/path/a\.nc\?versionId=v1 \(appears 2 times\)"
    )

    with pytest.raises(ValueError, match=expected_error_regex):
        extract.fn(object_references=object_references, fetcher=MagicMock())
