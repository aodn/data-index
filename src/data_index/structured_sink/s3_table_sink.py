from __future__ import annotations

import dataclasses
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
from pyiceberg.table import Table

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.structured_metadata import StructuredMetadata

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5


class StructuredS3TableSink(pydantic.BaseModel):
    """StructuredSink implementation that upserts Structured Metadata to a pre-provisioned Iceberg table.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Writes upsert on `s3_uri` (latest row wins within a batch and across writes).
    Concurrent upserts from multiple workers are safe — OCC conflicts are retried with
    exponential backoff.
    """

    type: typing.Literal["s3_table_sink"] = pydantic.Field(default="s3_table_sink")

    _instances: dict = pydantic.PrivateAttr(default_factory=lambda: {})

    iceberg_table_config: IcebergTableConfig

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
        """Create the namespace and table if they don't already exist.

        Pass a partition_spec to control partitioning (e.g. identity on collection).
        Defaults to no partitioning.
        """
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
                schema=StructuredMetadata.as_pyiceberg_schema(),
                partition_spec=partition_spec or PartitionSpec(),
            )
        except TableAlreadyExistsError:
            pass
        self._evolve_schema()

    def write(self, data: list[StructuredMetadata]) -> None:
        if not data:
            return
        arrow_table = self._to_arrow(self._dedupe_by_s3_uri(data))
        for attempt in range(_MAX_RETRIES):
            try:
                self.table.upsert(arrow_table, join_cols=["s3_uri"])
                return
            except CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                self.table.refresh()
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    def _evolve_schema(self) -> None:
        desired_schema = StructuredMetadata.as_pyiceberg_schema()
        for attempt in range(_MAX_RETRIES):
            try:
                self.table.update_schema().union_by_name(desired_schema).commit()
                return
            except CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                self.table.refresh()
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    @staticmethod
    def _to_arrow(data: list[StructuredMetadata]) -> pa.Table:
        return pa.Table.from_pylist(
            [dataclasses.asdict(row) for row in data],
            schema=StructuredMetadata.as_pyarrow_schema(),
        )

    @staticmethod
    def _dedupe_by_s3_uri(data: list[StructuredMetadata]) -> list[StructuredMetadata]:
        deduped: dict[str, StructuredMetadata] = {}
        for row in data:
            deduped[row.s3_uri] = row
        return list(deduped.values())
