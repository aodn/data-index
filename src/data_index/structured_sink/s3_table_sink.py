from __future__ import annotations

import dataclasses
import random
import time

import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.exceptions import (
    CommitFailedException,
    NamespaceAlreadyExistsError,
    TableAlreadyExistsError,
)
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, NestedField, StringType

from data_index.protocols import StructuredMetadata

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5


class StructuredS3TableSink:
    """StructuredSink implementation that appends Structured Metadata to a pre-provisioned Iceberg table.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Concurrent appends from multiple workers are safe — OCC conflicts are retried with
    exponential backoff.
    """

    ICEBERG_SCHEMA = Schema(
        NestedField(field_id=1, name="s3_uri", field_type=StringType(), required=True),
        NestedField(
            field_id=2, name="lat_min", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=3, name="lat_max", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=4, name="lon_min", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=5, name="lon_max", field_type=DoubleType(), required=False
        ),
        NestedField(
            field_id=6, name="time_min", field_type=StringType(), required=False
        ),
        NestedField(
            field_id=7, name="time_max", field_type=StringType(), required=False
        ),
        NestedField(field_id=8, name="crs", field_type=StringType(), required=False),
        NestedField(
            field_id=9, name="file_format", field_type=StringType(), required=False
        ),
        NestedField(
            field_id=10, name="collection", field_type=StringType(), required=False
        ),
    )

    def __init__(self, catalog: Catalog, namespace: str, table_name: str) -> None:
        self._catalog = catalog
        self._namespace = namespace
        self._table_name = table_name

    def provision(self, partition_spec: PartitionSpec | None = None) -> None:
        """Create the namespace and table if they don't already exist.

        Pass a partition_spec to control partitioning (e.g. identity on collection).
        Defaults to no partitioning.
        """
        try:
            self._catalog.create_namespace(self._namespace)
        except NamespaceAlreadyExistsError:
            pass
        try:
            self._catalog.create_table(
                identifier=(self._namespace, self._table_name),
                schema=self.ICEBERG_SCHEMA,
                partition_spec=partition_spec or PartitionSpec(),
            )
        except TableAlreadyExistsError:
            pass

    def write(self, data: list[StructuredMetadata]) -> None:
        if not data:
            return
        table = self._catalog.load_table((self._namespace, self._table_name))
        arrow_table = self._to_arrow(data)
        for attempt in range(_MAX_RETRIES):
            try:
                table.append(arrow_table)
                return
            except CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                table.refresh()
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    @staticmethod
    def _to_arrow(data: list[StructuredMetadata]) -> pa.Table:
        rows = [dataclasses.asdict(row) for row in data]
        schema = pa.schema(
            [
                pa.field("s3_uri", pa.string(), nullable=False),
                pa.field("lat_min", pa.float64(), nullable=True),
                pa.field("lat_max", pa.float64(), nullable=True),
                pa.field("lon_min", pa.float64(), nullable=True),
                pa.field("lon_max", pa.float64(), nullable=True),
                pa.field("time_min", pa.string(), nullable=True),
                pa.field("time_max", pa.string(), nullable=True),
                pa.field("crs", pa.string(), nullable=True),
                pa.field("file_format", pa.string(), nullable=True),
                pa.field("collection", pa.string(), nullable=True),
            ]
        )
        return pa.table(
            {
                "s3_uri": pa.array([r["s3_uri"] for r in rows], type=pa.string()),
                "lat_min": pa.array([r["lat_min"] for r in rows], type=pa.float64()),
                "lat_max": pa.array([r["lat_max"] for r in rows], type=pa.float64()),
                "lon_min": pa.array([r["lon_min"] for r in rows], type=pa.float64()),
                "lon_max": pa.array([r["lon_max"] for r in rows], type=pa.float64()),
                "time_min": pa.array([r["time_min"] for r in rows], type=pa.string()),
                "time_max": pa.array([r["time_max"] for r in rows], type=pa.string()),
                "crs": pa.array([r["crs"] for r in rows], type=pa.string()),
                "file_format": pa.array(
                    [r["file_format"] for r in rows], type=pa.string()
                ),
                "collection": pa.array(
                    [r["collection"] for r in rows], type=pa.string()
                ),
            },
            schema=schema,
        )
