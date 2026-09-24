from __future__ import annotations

import time
import typing

import boto3
import botocore.exceptions
import pydantic
from boto3.dynamodb.types import TypeSerializer

import data_index.protocols
import data_index.schema.metadata


class DynamoDBSink(pydantic.BaseModel):
    """DynamoDB sink for metadata rows keyed by query keys.

    Rows are written with `PutRequest` using materialized query keys:
    - `query_pk`: logical partition key (for example facility)
    - `query_sk`: composed lexicographic sort key (for example bucket/key)

    Note:
        If both Structured and Unstructured rows are sent to DynamoDB, they should
        target separate tables; both row types share the same object identity and
        would overwrite each other in a shared table.
    """

    type: typing.Literal["dynamodb_sink"] = pydantic.Field(
        default="dynamodb_sink",
        description="Discriminator used for runtime sink deserialization.",
    )
    table_name: str = pydantic.Field(
        min_length=3,
        max_length=255,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description=(
            "Target DynamoDB table name for metadata rows. The table is "
            "provisioned with query keys (`query_pk` as HASH and `query_sk` as RANGE)."
        ),
    )
    query_partition_field: str = pydantic.Field(
        default="facility",
        min_length=1,
        description=(
            "Row field used as the logical query partition key "
            "(for example `facility`)."
        ),
    )
    query_sort_fields: tuple[str, ...] = pydantic.Field(
        default=("bucket", "key"),
        description=(
            "Ordered row fields composed into the query sort key "
            "(for example `bucket`, `key`)."
        ),
    )
    query_sort_field_tags: tuple[str, ...] = pydantic.Field(
        default=(),
        description=(
            "Optional segment tags aligned to `query_sort_fields` used to build "
            "a tagged sort key (for example `BUCKET#...|KEY#...|SEQ#...`)."
        ),
    )
    query_partition_key_name: str = pydantic.Field(
        default="query_pk",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Materialized attribute name used as the table HASH key.",
    )
    query_sort_key_name: str = pydantic.Field(
        default="query_sk",
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Materialized attribute name used as the table RANGE key.",
    )
    query_sort_delimiter: str = pydantic.Field(
        default="|",
        min_length=1,
        max_length=8,
        description=(
            "Delimiter used when composing `query_sort_fields` into one "
            "lexicographically sortable string key."
        ),
    )
    region_name: str | None = pydantic.Field(
        default="ap-southeast-2",
        description=(
            "AWS region for the DynamoDB client. Defaults to the project's primary "
            "region (`ap-southeast-2`)."
        ),
    )
    endpoint_url: str | None = pydantic.Field(
        default=None,
        description=(
            "Optional DynamoDB endpoint override for local/test environments "
            "(for example LocalStack)."
        ),
    )
    billing_mode: typing.Literal["PAY_PER_REQUEST", "PROVISIONED"] = pydantic.Field(
        default="PAY_PER_REQUEST",
        description=(
            "DynamoDB billing mode used when creating the table if it does not "
            "already exist."
        ),
    )
    read_capacity_units: int = pydantic.Field(
        default=5,
        ge=1,
        description=(
            "Read capacity units for table creation when `billing_mode` is "
            "`PROVISIONED`. Ignored in `PAY_PER_REQUEST` mode."
        ),
    )
    write_capacity_units: int = pydantic.Field(
        default=5,
        ge=1,
        description=(
            "Write capacity units for table creation when `billing_mode` is "
            "`PROVISIONED`. Ignored in `PAY_PER_REQUEST` mode."
        ),
    )
    max_retries: int = pydantic.Field(
        default=5,
        ge=1,
        description=(
            "Maximum attempts for retryable write paths "
            "(for example unprocessed DynamoDB batch write items)."
        ),
    )
    base_backoff_seconds: float = pydantic.Field(
        default=0.5,
        gt=0,
        description="Base exponential backoff delay in seconds between retry attempts.",
    )
    batch_write_limit: int = pydantic.Field(
        default=25,
        ge=1,
        le=25,
        description=(
            "Maximum number of items per DynamoDB batch write request. DynamoDB "
            "service limit is 25."
        ),
    )

    @pydantic.model_validator(mode="after")
    def _validate_query_index_config(self) -> typing.Self:
        if not self.query_partition_field.strip():
            raise ValueError("query_partition_field must be a non-empty string")

        if not self.query_sort_fields:
            raise ValueError("query_sort_fields must contain at least one field")

        if self.query_partition_key_name == "hash":
            raise ValueError("query_partition_key_name cannot be `hash`")

        if self.query_sort_key_name == "hash":
            raise ValueError("query_sort_key_name cannot be `hash`")

        if self.query_partition_key_name == self.query_sort_key_name:
            raise ValueError(
                "query_partition_key_name and query_sort_key_name must be different"
            )

        if any(not field.strip() for field in self.query_sort_fields):
            raise ValueError("query_sort_fields cannot contain empty field names")

        if self.query_sort_field_tags and len(self.query_sort_field_tags) != len(
            self.query_sort_fields
        ):
            raise ValueError(
                "query_sort_field_tags must be empty or match query_sort_fields length"
            )

        if any(not tag.strip() for tag in self.query_sort_field_tags):
            raise ValueError("query_sort_field_tags cannot contain empty tag values")

        return self

    @property
    def client(self):
        """Build a DynamoDB client from sink configuration.

        A fresh client is inexpensive and keeps config source-of-truth on the model,
        which is useful when sinks are deserialized in remote workers.
        """
        return boto3.client(
            "dynamodb",
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )

    def _inject_query_keys(self, row_item: dict[str, typing.Any]) -> None:
        partition_field = self.query_partition_field
        partition_value = row_item.get(partition_field)
        if not isinstance(partition_value, str) or not partition_value:
            raise ValueError(
                f"Expected non-empty string for query partition field `{partition_field}`"
            )

        sort_parts: list[str] = []
        for index, field_name in enumerate(self.query_sort_fields):
            if field_name not in row_item:
                raise ValueError(
                    f"Missing query sort field `{field_name}` in row payload"
                )
            field_value = row_item[field_name]
            if field_value is None:
                raise ValueError(
                    f"Query sort field `{field_name}` cannot be None in row payload"
                )
            value_str = str(field_value)
            if self.query_sort_field_tags:
                sort_parts.append(f"{self.query_sort_field_tags[index]}#{value_str}")
            else:
                sort_parts.append(value_str)

        sort_key = self.query_sort_delimiter.join(sort_parts)
        row_item[self.query_partition_key_name] = partition_value
        row_item[self.query_sort_key_name] = sort_key

    def provision(self) -> None:
        """Ensure the sink table exists before writes.

        The flow is intentionally two-phase:
        1. Fast path: `describe_table` returns, so we reuse existing schema.
        2. Create path: only `ResourceNotFoundException` triggers table creation.

        Other AWS errors are re-raised immediately to avoid silently masking
        permission/configuration failures.
        """
        try:
            self.client.describe_table(TableName=self.table_name)
            return
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] != "ResourceNotFoundException":
                raise

        attribute_definitions: dict[str, str] = {
            self.query_partition_key_name: "S",
            self.query_sort_key_name: "S",
        }
        key_schema: list[dict[str, str]] = [
            {"AttributeName": self.query_partition_key_name, "KeyType": "HASH"},
            {"AttributeName": self.query_sort_key_name, "KeyType": "RANGE"},
        ]

        create_table_args: dict[str, typing.Any] = {
            "TableName": self.table_name,
            "KeySchema": key_schema,
            "AttributeDefinitions": [
                {"AttributeName": name, "AttributeType": attr_type}
                for name, attr_type in attribute_definitions.items()
            ],
            "BillingMode": self.billing_mode,
        }

        if self.billing_mode == "PROVISIONED":
            create_table_args["ProvisionedThroughput"] = {
                "ReadCapacityUnits": self.read_capacity_units,
                "WriteCapacityUnits": self.write_capacity_units,
            }

        self.client.create_table(**create_table_args)
        waiter = self.client.get_waiter("table_exists")
        waiter.wait(TableName=self.table_name)

    def _serialize_item(
        self,
        row: data_index.schema.metadata.StructuredMetadata
        | data_index.schema.metadata.UnstructuredMetadata,
        serializer: TypeSerializer,
    ) -> dict[str, dict[str, typing.Any]]:
        """Serialize one metadata row to DynamoDB's typed-attribute map."""
        # Preserve metadata normalization rules from Schema.as_dynamodb_type_spec:
        # float coercion, null handling, and list/map JSON stringification.
        row_item = row.as_dynamodb_item()
        self._inject_query_keys(row_item)
        return {key: serializer.serialize(value) for key, value in row_item.items()}

    def _serialize_rows(
        self,
        metadata: list[
            data_index.schema.metadata.StructuredMetadata
            | data_index.schema.metadata.UnstructuredMetadata
        ],
        serializer: TypeSerializer,
    ) -> list[dict[str, dict[str, typing.Any]]]:
        """Serialize metadata rows using a single, normalized row contract."""
        return [
            self._serialize_item(row=row, serializer=serializer) for row in metadata
        ]

    def _batch_put(self, items: list[dict[str, dict[str, typing.Any]]]) -> None:
        """Write one DynamoDB batch with bounded retries.

        DynamoDB may return `UnprocessedItems` during throttling/backpressure.
        Those items are safe to retry because writes are idempotent by primary key.
        """
        request_items = {
            self.table_name: [{"PutRequest": {"Item": item}} for item in items]
        }
        for attempt in range(self.max_retries):
            response = self.client.batch_write_item(RequestItems=request_items)
            unprocessed_items = response.get("UnprocessedItems", {})
            pending = unprocessed_items.get(self.table_name, [])
            if not pending:
                return
            request_items = {self.table_name: pending}
            if attempt == self.max_retries - 1:
                raise RuntimeError(
                    f"Failed to write {len(pending)} DynamoDB items after {self.max_retries} attempts"
                )
            time.sleep(self.base_backoff_seconds * (2**attempt))

    def write(
        self,
        metadata: list[data_index.schema.metadata.StructuredMetadata]
        | list[data_index.schema.metadata.UnstructuredMetadata]
        | list[data_index.protocols.DeadLetter],
    ) -> None:
        """Persist structured or unstructured rows to DynamoDB.

        `MetadataSink` protocol also allows dead-letter rows, but this sink
        intentionally rejects them to keep dead-letter retention isolated from
        metadata tables and key semantics.
        """
        if not metadata:
            return

        if not all(
            isinstance(
                row,
                (
                    data_index.schema.metadata.StructuredMetadata,
                    data_index.schema.metadata.UnstructuredMetadata,
                ),
            )
            for row in metadata
        ):
            raise TypeError(
                "DynamoDBSink only accepts StructuredMetadata and UnstructuredMetadata rows"
            )

        serializer = TypeSerializer()
        items = self._serialize_rows(metadata=metadata, serializer=serializer)

        for i in range(0, len(items), self.batch_write_limit):
            self._batch_put(items=items[i : i + self.batch_write_limit])
