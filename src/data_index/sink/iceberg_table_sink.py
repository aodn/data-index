from __future__ import annotations

import random
import time
import typing

import pydantic
import pyiceberg.catalog
import pyiceberg.exceptions
import pyiceberg.partitioning
import pyiceberg.table
import pyiceberg.transforms

import data_index.iceberg_config
import data_index.protocols
import data_index.schema.metadata

_MAX_RETRIES = 5
_BASE_BACKOFF = 0.5


class IcebergTableSink(pydantic.BaseModel):
    """Unified IcebergTableSink implementation for upserting metadata and dead letters.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Concurrent upserts from multiple workers are safe — OCC conflicts are retried with
    exponential backoff.
    """

    type: typing.Literal["iceberg_table_sink"] = pydantic.Field(
        default="iceberg_table_sink"
    )
    schema_kind: typing.Literal["structured", "unstructured", "dead_letter"]
    iceberg_table_config: data_index.iceberg_config.IcebergTableConfig
    partition_column: str | None = pydantic.Field(default=None)

    @property
    def catalog(self) -> pyiceberg.catalog.Catalog:
        return self.iceberg_table_config.catalog_config.build()

    @property
    def table(self) -> pyiceberg.table.Table:
        return self.iceberg_table_config.load()

    @property
    def _metadata_cls(
        self,
    ) -> (
        data_index.schema.metadata.StructuredMetadata
        | data_index.schema.metadata.UnstructuredMetadata
    ):
        """Dynamically resolves the target metadata class wrapper based on kind."""
        match self.schema_kind:
            case "structured":
                return data_index.schema.metadata.StructuredMetadata
            case "unstructured":
                return data_index.schema.metadata.UnstructuredMetadata
            case "dead_letter":
                return data_index.protocols.DeadLetter
            case _:
                raise ValueError(f"unsupported metadata_kind: {self.metadata_kind}")

    def provision(
        self,
        reset: bool = False,
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
        except pyiceberg.exceptions.NamespaceAlreadyExistsError:
            pass

        if reset:
            try:
                self.catalog.drop_table(identifier)
            except pyiceberg.exceptions.NoSuchTableError:
                pass

        try:
            self.catalog.create_table(
                identifier=identifier,
                schema=self._metadata_cls.as_pyiceberg_schema(),
                partition_spec=self._partition_spec()
                if self.partition_column
                else pyiceberg.partitioning.UNPARTITIONED_PARTITION_SPEC,
            )
        except pyiceberg.exceptions.TableAlreadyExistsError:
            pass

        self._evolve_schema()

    def _evolve_schema(self) -> None:
        desired_schema = self._metadata_cls.as_pyiceberg_schema()
        for attempt in range(_MAX_RETRIES):
            try:
                self.table.update_schema().union_by_name(desired_schema).commit()
                return
            except pyiceberg.exceptions.CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    def _partition_spec(self) -> pyiceberg.partitioning.PartitionSpec:
        facility_field_id = (
            self._metadata_cls.as_pyiceberg_schema()
            .find_field(self.partition_column)
            .field_id
        )
        return pyiceberg.partitioning.PartitionSpec(
            pyiceberg.partitioning.PartitionField(
                source_id=facility_field_id,
                field_id=1000,
                transform=pyiceberg.transforms.IdentityTransform(),
                name=self.partition_column,
            )
        )

    def write(
        self,
        metadata: list[
            data_index.schema.metadata.StructuredMetadata
            | data_index.schema.metadata.UnstructuredMetadata
        ],
    ) -> None:

        # Fixed: passing extracted_objects positionally to avoid signature mismatch
        table = self._metadata_cls.to_arrow(metadata=metadata)

        for attempt in range(_MAX_RETRIES):
            try:
                self.table.upsert(df=table, join_cols=["hash"])
                return []
            except pyiceberg.exceptions.CommitFailedException:
                if attempt == _MAX_RETRIES - 1:
                    raise
                time.sleep(
                    random.uniform(a=_BASE_BACKOFF, b=_BASE_BACKOFF * 2) * (2**attempt)
                )
