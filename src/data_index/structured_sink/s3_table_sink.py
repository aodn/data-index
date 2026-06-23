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
    NoSuchTableError,
    TableAlreadyExistsError,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.table import Table
from pyiceberg.transforms import IdentityTransform

from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.schema.metadata import StructuredMetadata

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5
_SCHEMA_VERSION_PROPERTY = "schema_version"


class StructuredS3TableSink(pydantic.BaseModel):
    """StructuredSink implementation that upserts Structured Metadata to a pre-provisioned Iceberg table.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Writes upsert on (`bucket`, `key`, `version_id`) (latest row wins within
    a batch and across writes).
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

    def provision(
        self, partition_spec: PartitionSpec | None = None, reset: bool = False
    ) -> None:
        """Create the namespace and table if they don't already exist.

        Pass a partition_spec to override default partitioning.
        By default rows are partitioned by facility and time_coverage_start_year.
        Set reset=True to drop and recreate the table (explicit opt-in destructive mode).
        """
        identifier = (
            self.iceberg_table_config.namespace,
            self.iceberg_table_config.table_name,
        )
        try:
            self.catalog.create_namespace(self.iceberg_table_config.namespace)
        except NamespaceAlreadyExistsError:
            pass
        if reset:
            try:
                self.catalog.drop_table(identifier)
            except NoSuchTableError:
                pass
            self._instances.pop("table", None)
        try:
            self.catalog.create_table(
                identifier=identifier,
                schema=StructuredMetadata.as_pyiceberg_schema(),
                partition_spec=partition_spec or self._default_partition_spec(),
                properties={
                    _SCHEMA_VERSION_PROPERTY: str(StructuredMetadata.SCHEMA_VERSION)
                },
            )
        except TableAlreadyExistsError:
            pass
        self._evolve_schema()
        self._ensure_schema_version_property()

    def write(self, data: list[StructuredMetadata]) -> None:
        if not data:
            return
        arrow_table = self._to_arrow(data=data)
        for attempt in range(_MAX_RETRIES):
            try:
                self.table.upsert(arrow_table, join_cols=["hash"])
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

    def _ensure_schema_version_property(self) -> None:
        expected = str(StructuredMetadata.SCHEMA_VERSION)
        for attempt in range(_MAX_RETRIES):
            if self.table.properties.get(_SCHEMA_VERSION_PROPERTY) == expected:
                return
            try:
                self.table.transaction().set_properties(
                    **{_SCHEMA_VERSION_PROPERTY: expected}
                ).commit_transaction()
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
    def _default_partition_spec() -> PartitionSpec:
        schema = StructuredMetadata.as_pyiceberg_schema()
        facility_field_id = schema.find_field("facility").field_id
        return PartitionSpec(
            PartitionField(
                source_id=facility_field_id,
                field_id=1000,
                transform=IdentityTransform(),
                name="facility",
            ),
        )

    @staticmethod
    def _dedupe_by_identity(data: list[StructuredMetadata]) -> list[StructuredMetadata]:
        deduped: dict[tuple[str, str, str], StructuredMetadata] = {}
        for row in data:
            deduped[(row.bucket, row.key, row.version_id)] = row
        return list(deduped.values())
