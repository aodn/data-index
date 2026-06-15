from __future__ import annotations

import json
import random
import time
import typing

import pyarrow as pa
import pydantic
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import (
    CommitFailedException,
    NamespaceAlreadyExistsError,
    TableAlreadyExistsError,
)
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.types import NestedField, StringType

from data_index._collection import derive_collection
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5

_ARROW_SCHEMA = pa.schema(
    [
        pa.field("s3_uri", pa.string(), nullable=False),
        pa.field("collection", pa.string(), nullable=True),
        pa.field("metadata", pa.string(), nullable=True),
    ]
)


class UnstructuredS3TableSink(pydantic.BaseModel):
    """UnstructuredSink implementation that appends Unstructured Metadata to a pre-provisioned
    Iceberg table with schema (s3_uri, collection, metadata).

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Collection is derived from the s3_uri key at write time (second path segment).
    Metadata is JSON-encoded. Concurrent appends from multiple workers are safe —
    OCC conflicts are retried with exponential backoff.
    """

    type: typing.Literal["s3_table_sink"] = pydantic.Field(default="s3_table_sink")

    _ICEBERG_SCHEMA: Schema = pydantic.PrivateAttr(
        default_factory=lambda: Schema(
            NestedField(
                field_id=1, name="s3_uri", field_type=StringType(), required=True
            ),
            NestedField(
                field_id=2, name="collection", field_type=StringType(), required=False
            ),
            NestedField(
                field_id=3, name="metadata", field_type=StringType(), required=False
            ),
        )
    )

    iceberg_table_config: IcebergTableConfig
    _instances: dict = pydantic.PrivateAttr(default_factory=lambda: {})

    @property
    def catalog(self) -> Catalog:
        return self._instances.get(
            "catalog",
            self.iceberg_table_config.catalog_config.build(),
        )

    @property
    def table(self) -> Table:
        return self._instances.get(
            "table",
            self.iceberg_table_config.load(),
        )

    def provision(self, partition_spec: PartitionSpec | None = None) -> None:
        """Create the namespace and table if they don't already exist."""
        try:
            self.catalog.create_namespace(self.iceberg_table_config.namespace)
        except NamespaceAlreadyExistsError:
            pass
        try:
            self.catalog.create_table(
                identifier=(
                    self.iceberg_table_config.namespace,
                    self.iceberg_table_config.table_name,
                ),
                schema=self._ICEBERG_SCHEMA,
                partition_spec=partition_spec or PartitionSpec(),
            )
        except TableAlreadyExistsError:
            pass

    def write(self, data: dict[str, dict]) -> None:
        if not data:
            return
        arrow_table = self._to_arrow(data)
        for attempt in range(_MAX_RETRIES):
            try:
                self.table.append(arrow_table)
                return
            except CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                self.table.refresh()
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    @staticmethod
    def _to_arrow(data: dict[str, dict]) -> pa.Table:
        uris = list(data.keys())
        return pa.table(
            {
                "s3_uri": pa.array(uris, type=pa.string()),
                "collection": pa.array(
                    [derive_collection(u) for u in uris], type=pa.string()
                ),
                "metadata": pa.array(
                    [json.dumps(data[u]) for u in uris], type=pa.string()
                ),
            },
            schema=_ARROW_SCHEMA,
        )
