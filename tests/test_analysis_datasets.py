from unittest.mock import MagicMock, patch

from data_index.analysis.datasets import get_dataset_arrow_dataset


def test_get_dataset_arrow_dataset_builds_anonymous_s3_parquet_dataset():
    s3_filesystem = MagicMock()
    schema = MagicMock()
    arrow_dataset = MagicMock()

    with (
        patch(
            "data_index.analysis.datasets.pyarrow.fs.S3FileSystem",
            return_value=s3_filesystem,
        ) as create_s3_filesystem,
        patch(
            "data_index.analysis.datasets.pyarrow.parquet.read_schema",
            return_value=schema,
        ) as read_schema,
        patch(
            "data_index.analysis.datasets.pyarrow.dataset.dataset",
            return_value=arrow_dataset,
        ) as create_dataset,
    ):
        result = get_dataset_arrow_dataset(
            dataset="ocean_glider_delayed_qc",
            bucket="custom-bucket",
        )

    assert result is arrow_dataset
    create_s3_filesystem.assert_called_once_with(
        region="ap-southeast-2",
        anonymous=True,
    )
    read_schema.assert_called_once_with(
        where="custom-bucket/ocean_glider_delayed_qc.parquet/_common_metadata",
        filesystem=s3_filesystem,
    )
    create_dataset.assert_called_once_with(
        source="custom-bucket/ocean_glider_delayed_qc.parquet",
        filesystem=s3_filesystem,
        schema=schema,
    )


def test_get_dataset_arrow_dataset_builds_signed_s3_parquet_dataset():
    session = MagicMock()
    credentials = MagicMock()
    frozen_credentials = MagicMock()
    s3_filesystem = MagicMock()
    schema = MagicMock()
    arrow_dataset = MagicMock()

    session.get_credentials.return_value = credentials
    credentials.get_frozen_credentials.return_value = frozen_credentials
    frozen_credentials.access_key = "access"
    frozen_credentials.secret_key = "secret"
    frozen_credentials.token = "session-token"

    with (
        patch(
            "data_index.analysis.datasets.boto3.Session", return_value=session
        ) as create_session,
        patch(
            "data_index.analysis.datasets.pyarrow.fs.S3FileSystem",
            return_value=s3_filesystem,
        ) as create_s3_filesystem,
        patch(
            "data_index.analysis.datasets.pyarrow.parquet.read_schema",
            return_value=schema,
        ) as read_schema,
        patch(
            "data_index.analysis.datasets.pyarrow.dataset.dataset",
            return_value=arrow_dataset,
        ) as create_dataset,
    ):
        result = get_dataset_arrow_dataset(
            dataset="ocean_glider_delayed_qc",
            bucket="custom-bucket",
            skip_signature=False,
            profile_name="custom-profile",
        )

    assert result is arrow_dataset
    create_session.assert_called_once_with(
        profile_name="custom-profile",
        region_name="ap-southeast-2",
    )
    create_s3_filesystem.assert_called_once_with(
        region="ap-southeast-2",
        access_key="access",
        secret_key="secret",
        session_token="session-token",
    )
    read_schema.assert_called_once_with(
        where="custom-bucket/ocean_glider_delayed_qc.parquet/_common_metadata",
        filesystem=s3_filesystem,
    )
    create_dataset.assert_called_once_with(
        source="custom-bucket/ocean_glider_delayed_qc.parquet",
        filesystem=s3_filesystem,
        schema=schema,
    )
