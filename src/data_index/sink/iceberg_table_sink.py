from __future__ import annotations

import dataclasses
import random
import time
import typing

import pyarrow as pa
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
    """Unified S3TableSink implementation that upserts either Unstructured or Structured
    Metadata to a pre-provisioned Iceberg table.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Concurrent upserts from multiple workers are safe — OCC conflicts are retried with
    exponential backoff.
    """

    type: typing.Literal["s3_table_sink"] = pydantic.Field(default="s3_table_sink")
    metadata_kind: typing.Literal["structured", "unstructured"] = pydantic.Field(
        default="structured"
    )
    iceberg_table_config: data_index.iceberg_config.IcebergTableConfig

    _instances: dict = pydantic.PrivateAttr(default_factory=lambda: {})

    @property
    def catalog(self) -> pyiceberg.catalog.Catalog:
        return self._instances.get(
            "catalog",
            self.iceberg_table_config.catalog_config.build(),
        )

    @property
    def table(self) -> pyiceberg.table.Table:
        return self._instances.get(
            "table",
            self.iceberg_table_config.load(),
        )

    @property
    def _metadata_cls(self) -> typing.Type:
        """Dynamically resolves the target metadata class wrapper based on kind."""
        match self.metadata_kind:
            case "structured":
                return data_index.schema.metadata.StructuredMetadata
            case "unstructured":
                return data_index.schema.metadata.UnstructuredMetadata
            case _:
                raise ValueError(f"unsupported metadata_kind: {self.metadata_kind}")

    def provision(
        self,
        partition_spec: pyiceberg.partitioning.PartitionSpec | None = None,
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
            self._instances.pop("table", None)

        try:
            self.catalog.create_table(
                identifier=identifier,
                schema=self._metadata_cls.as_pyiceberg_schema(),
                partition_spec=partition_spec or self._default_partition_spec(),
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
                self.table.refresh()
                time.sleep(
                    random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                )

    def _default_partition_spec(self) -> pyiceberg.partitioning.PartitionSpec:
        facility_field_id = (
            self._metadata_cls.as_pyiceberg_schema().find_field("facility").field_id
        )
        return pyiceberg.partitioning.PartitionSpec(
            pyiceberg.partitioning.PartitionField(
                source_id=facility_field_id,
                field_id=1000,
                transform=pyiceberg.transforms.IdentityTransform(),
                name="facility",
            )
        )

    def _to_arrow(
        self, extracted_objects: list[data_index.protocols.ExtractedObject]
    ) -> pa.Table:
        # Dynamically switch between extraction_result.unstructured_metadata and structured_metadata
        metadata_attr = f"{self.metadata_kind}_metadata"

        return pa.Table.from_pylist(
            [
                dataclasses.asdict(getattr(obj.extraction_result, metadata_attr))
                for obj in extracted_objects
            ],
            schema=self._metadata_cls.as_pyarrow_schema(),
        )

    def write(
        self, extracted_objects: list[data_index.protocols.ExtractedObject]
    ) -> list[data_index.protocols.DeadLetter]:
        try:
            if not extracted_objects:
                return list()

            # Fixed: passing extracted_objects positionally to avoid signature mismatch
            arrow_table = self._to_arrow(extracted_objects)

            for attempt in range(_MAX_RETRIES):
                try:
                    self.table.upsert(arrow_table, join_cols=["hash"])
                    return list()
                except pyiceberg.exceptions.CommitFailedException:
                    if attempt == _MAX_RETRIES - 1:
                        raise
                    self.table.refresh()
                    time.sleep(
                        random.uniform(_BASE_BACKOFF, _BASE_BACKOFF * 2) * (2**attempt)
                    )

        # Return dead letter for every extracted object if execution failed
        except Exception as e:
            return [
                data_index.protocols.DeadLetter.from_object_reference(
                    object_reference=extracted_object.object_reference, error=str(e)
                )
                for extracted_object in extracted_objects
            ]
