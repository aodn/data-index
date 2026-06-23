from __future__ import annotations

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
from data_index.schema.metadata import UnstructuredMetadata

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5
_SCHEMA_VERSION_PROPERTY = "schema_version"
_SCHEMA_VERSION = 1


class UnstructuredS3TableSink(pydantic.BaseModel):
    """UnstructuredSink implementation that upserts Unstructured Metadata to a pre-provisioned
    Iceberg table with schema (bucket, key, version_id, facility, metadata).

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Facility is derived from key at write time (second path segment).
    Metadata is JSON-encoded. Writes upsert on (`bucket`, `key`, `version_id`).
    Concurrent upserts from multiple workers are safe —
    OCC conflicts are retried with exponential backoff.
    """

    type: typing.Literal["s3_table_sink"] = pydantic.Field(default="s3_table_sink")

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

    def provision(
        self, partition_spec: PartitionSpec | None = None, reset: bool = False
    ) -> None:
        """Create the namespace and table if they don't already exist.

        Pass a partition_spec to override default partitioning.
        By default rows are partitioned by facility.
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
                schema=UnstructuredMetadata.as_pyiceberg_schema(),
                partition_spec=partition_spec or self._default_partition_spec(),
                properties={_SCHEMA_VERSION_PROPERTY: str(_SCHEMA_VERSION)},
            )
        except TableAlreadyExistsError:
            pass
        self._evolve_schema()
        self._ensure_schema_version_property()

    def _evolve_schema(self) -> None:
        desired_schema = UnstructuredMetadata.as_pyiceberg_schema()
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
        expected = str(_SCHEMA_VERSION)
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

    def _default_partition_spec(self) -> PartitionSpec:
        facility_field_id = (
            UnstructuredMetadata.as_pyiceberg_schema().find_field("facility").field_id
        )
        return PartitionSpec(
            PartitionField(
                source_id=facility_field_id,
                field_id=1000,
                transform=IdentityTransform(),
                name="facility",
            )
        )

    @staticmethod
    def _to_arrow(unstructured_metadata: list[UnstructuredMetadata]) -> pa.Table:

        # Construct table
        return pa.table(
            {
                "bucket": pa.array(
                    [
                        unstructured_metadata.bucket
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "key": pa.array(
                    [
                        unstructured_metadata.key
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "version_id": pa.array(
                    [
                        unstructured_metadata.version_id
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "hash": pa.array(
                    [
                        unstructured_metadata.hash
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "schema_version": pa.repeat(
                    pa.scalar(UnstructuredMetadata.SCHEMA_VERSION, type=pa.int64()),
                    len(unstructured_metadata),
                ),
                "metadata": pa.array(
                    [
                        unstructured_metadata.metadata
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "file_format": pa.array(
                    [
                        unstructured_metadata.file_format
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
                "facility": pa.array(
                    [
                        unstructured_metadata.facility
                        for unstructured_metadata in unstructured_metadata
                    ],
                    type=pa.string(),
                ),
            },
            schema=UnstructuredMetadata.as_pyarrow_schema(),
        )

    def write(self, unstructured_metadata: list[UnstructuredMetadata]) -> None:

        # Return if no data
        if not unstructured_metadata:
            return

        # Turn into an arrow table
        arrow_table = self._to_arrow(unstructured_metadata=unstructured_metadata)
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
