from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

import data_index.protocols
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata
from data_index.sink import DynamoDBSink


def _unstructured_metadata(hash_value: str) -> UnstructuredMetadata:
    return UnstructuredMetadata(
        bucket="bucket",
        key=f"path/{hash_value}.nc",
        version_id="version",
        hash=hash_value,
        file_format="NETCDF4",
        facility="ANMN",
        metadata='{"global_attrs":{}}',
    )


def test_provision_creates_table_when_missing():
    mock_client = MagicMock()
    mock_client.describe_table.side_effect = botocore.exceptions.ClientError(
        error_response={
            "Error": {"Code": "ResourceNotFoundException", "Message": "table missing"}
        },
        operation_name="DescribeTable",
    )
    mock_waiter = MagicMock()
    mock_client.get_waiter.return_value = mock_waiter

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.provision()

    mock_client.create_table.assert_called_once_with(
        TableName="unstructured-metadata",
        KeySchema=[
            {"AttributeName": "query_pk", "KeyType": "HASH"},
            {"AttributeName": "query_sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "query_pk", "AttributeType": "S"},
            {"AttributeName": "query_sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    mock_waiter.wait.assert_called_once_with(TableName="unstructured-metadata")


def test_provision_noops_when_table_exists():
    mock_client = MagicMock()
    mock_client.describe_table.return_value = {"Table": {"TableName": "existing"}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.provision()

    mock_client.create_table.assert_not_called()


def test_provisioned_mode_uses_default_capacity_units():
    mock_client = MagicMock()
    mock_client.describe_table.side_effect = botocore.exceptions.ClientError(
        error_response={
            "Error": {"Code": "ResourceNotFoundException", "Message": "table missing"}
        },
        operation_name="DescribeTable",
    )
    sink = DynamoDBSink(
        table_name="unstructured-metadata",
        billing_mode="PROVISIONED",
    )

    mock_waiter = MagicMock()
    mock_client.get_waiter.return_value = mock_waiter

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink.provision()

    mock_client.create_table.assert_called_once_with(
        TableName="unstructured-metadata",
        KeySchema=[
            {"AttributeName": "query_pk", "KeyType": "HASH"},
            {"AttributeName": "query_sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "query_pk", "AttributeType": "S"},
            {"AttributeName": "query_sk", "AttributeType": "S"},
        ],
        BillingMode="PROVISIONED",
        ProvisionedThroughput={
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5,
        },
    )


def test_provision_supports_custom_query_key_attribute_names():
    mock_client = MagicMock()
    mock_client.describe_table.side_effect = botocore.exceptions.ClientError(
        error_response={
            "Error": {"Code": "ResourceNotFoundException", "Message": "table missing"}
        },
        operation_name="DescribeTable",
    )
    mock_waiter = MagicMock()
    mock_client.get_waiter.return_value = mock_waiter

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_field="facility",
            query_sort_fields=("bucket", "key", "version_id"),
            query_partition_key_name="facility_pk",
            query_sort_key_name="facility_sk",
        )
        sink.provision()

    mock_client.create_table.assert_called_once_with(
        TableName="unstructured-metadata",
        KeySchema=[
            {"AttributeName": "facility_pk", "KeyType": "HASH"},
            {"AttributeName": "facility_sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "facility_pk", "AttributeType": "S"},
            {"AttributeName": "facility_sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    mock_waiter.wait.assert_called_once_with(TableName="unstructured-metadata")


def test_query_sort_fields_must_not_be_empty():
    with pytest.raises(
        ValueError,
        match="query_sort_fields must contain at least one field",
    ):
        DynamoDBSink(
            table_name="unstructured-metadata",
            query_sort_fields=(),
        )


def test_query_partition_key_name_cannot_be_hash():
    with pytest.raises(
        ValueError,
        match="query_partition_key_name cannot be `hash`",
    ):
        DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_key_name="hash",
        )


def test_query_sort_key_name_cannot_be_hash():
    with pytest.raises(
        ValueError,
        match="query_sort_key_name cannot be `hash`",
    ):
        DynamoDBSink(
            table_name="unstructured-metadata",
            query_sort_key_name="hash",
        )


def test_query_sort_tags_must_align_with_sort_fields():
    with pytest.raises(
        ValueError,
        match="query_sort_field_tags must be empty or match query_sort_fields length",
    ):
        DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_field="facility",
            query_sort_fields=("bucket", "key", "version_id"),
            query_sort_field_tags=("BUCKET", "KEY"),
        )


def test_write_accepts_structured_rows():
    mock_client = MagicMock()
    structured_row = StructuredMetadata(
        bucket="bucket",
        key="key",
        version_id="version",
        hash="abc",
        file_format="NETCDF4",
        facility="ANMN",
        geospatial_lat_min=12.5,
    )
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.write(metadata=[structured_row])

    mock_client.batch_write_item.assert_called_once()
    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    assert item["geospatial_lat_min"] == {"N": "12.5"}


def test_write_adds_query_keys():
    mock_client = MagicMock()
    structured_row = StructuredMetadata(
        bucket="bucket",
        key="key",
        version_id="version",
        hash="abc",
        file_format="NETCDF4",
        facility="ANMN",
    )
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_field="facility",
            query_sort_fields=("bucket", "key", "version_id"),
        )
        sink.write(metadata=[structured_row])

    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    expected_sort_key = "bucket|key|version"
    assert item["query_pk"] == {"S": "ANMN"}
    assert item["query_sk"] == {"S": expected_sort_key}
    assert "row_kind" not in item


def test_write_unstructured_uses_same_serialization_contract():
    mock_client = MagicMock()
    unstructured_row = _unstructured_metadata(hash_value="h1")
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_field="facility",
            query_sort_fields=("bucket", "key", "version_id"),
        )
        sink.write(metadata=[unstructured_row])

    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    expected_sort_key = "bucket|path/h1.nc|version"
    assert item["query_pk"] == {"S": "ANMN"}
    assert item["query_sk"] == {"S": expected_sort_key}
    assert item["metadata"] == {"S": '{"global_attrs":{}}'}
    assert "row_kind" not in item


def test_write_adds_tagged_query_sort_key():
    mock_client = MagicMock()
    structured_row = StructuredMetadata(
        bucket="bucket",
        key="key",
        version_id="version",
        hash="abc",
        file_format="NETCDF4",
        facility="ANMN",
    )
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(
            table_name="unstructured-metadata",
            query_partition_field="facility",
            query_sort_fields=("bucket", "key", "version_id"),
            query_sort_field_tags=("BUCKET", "KEY", "SEQ"),
        )
        sink.write(metadata=[structured_row])

    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    assert item["query_sk"] == {"S": "BUCKET#bucket|KEY#key|SEQ#version"}


def test_write_serializes_structured_maps_as_string_attributes():
    mock_client = MagicMock()
    structured_row = StructuredMetadata(
        bucket="bucket",
        key="key",
        version_id="version",
        hash="abc",
        file_format="NETCDF4",
        facility="ANMN",
        dimension_sizes={"time": 10},
    )
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.write(metadata=[structured_row])

    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    assert item["dimension_sizes"] == {"S": '{"time":10}'}


def test_write_rejects_dead_letter_rows():
    mock_client = MagicMock()
    dead_letter = data_index.protocols.DeadLetter(
        bucket="bucket",
        key="key",
        version_id="version",
        size=1,
        hash="abc",
        error="failed",
    )

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        with pytest.raises(TypeError):
            sink.write(metadata=[dead_letter])


def test_write_coerces_non_finite_structured_floats_to_null():
    mock_client = MagicMock()
    structured_row = StructuredMetadata(
        bucket="bucket",
        key="key",
        version_id="version",
        hash="abc",
        file_format="NETCDF4",
        facility="ANMN",
        geospatial_lat_min=float("nan"),
    )
    mock_client.batch_write_item.return_value = {"UnprocessedItems": {}}

    with patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.write(metadata=[structured_row])

    request_items = mock_client.batch_write_item.call_args.kwargs["RequestItems"]
    item = request_items["unstructured-metadata"][0]["PutRequest"]["Item"]
    assert "geospatial_lat_min" not in item


def test_write_retries_unprocessed_items():
    mock_client = MagicMock()
    first_response = {
        "UnprocessedItems": {
            "unstructured-metadata": [
                {
                    "PutRequest": {
                        "Item": {
                            "query_pk": {"S": "ANMN"},
                            "query_sk": {"S": "bucket|path/h1.nc|version"},
                        }
                    }
                }
            ]
        }
    }
    second_response = {"UnprocessedItems": {}}
    mock_client.batch_write_item.side_effect = [first_response, second_response]

    with (
        patch("data_index.sink.dynamodb_sink.boto3.client", return_value=mock_client),
        patch("data_index.sink.dynamodb_sink.time.sleep"),
    ):
        sink = DynamoDBSink(table_name="unstructured-metadata")
        sink.write(metadata=[_unstructured_metadata(hash_value="h1")])

    assert mock_client.batch_write_item.call_count == 2
