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
_SCHEMA_VERSION_PROPERTY = "schema_version"
_UNSTRUCTURED_SCHEMA_VERSION = 1


class IcebergTableSink(pydantic.BaseModel):
    """Unified S3TableSink implementation that upserts either Unstructured or Structured
    Metadata to a pre-provisioned Iceberg table.

    The table must be created before writing — call provision() or use the Orchestrator's pre_run
    hook. Concurrent upserts from multiple workers are safe — OCC conflicts are retried with
    exponential backoff.
    """

    type: typing.Literal["s3_table_sink"] = pydantic.Field(default="s3_table_sink")
    metadata_kind: typing.Literal["unstructured", "structured"] = pydantic.Field(
        default="unstructured"
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
    def _metadata_cls(self) -> typing.Any:
        """Dynamically resolves the target metadata class wrapper based on kind."""
        match self.metadata_kind:
            case "unstructured":
                return data_index.schema.metadata.UnstructuredMetadata
            case "structured":
                return data_index.schema.metadata.StructuredMetadata
            case _:
                raise ValueError(f"unsupported metadata_kind: {self.metadata_kind}")

    @property
    def _schema_version(self) -> str:
        """Dynamically resolves the schema version string based on kind."""
        if self.metadata_kind == "unstructured":
            return str(_UNSTRUCTURED_SCHEMA_VERSION)
        return str(data_index.schema.metadata.StructuredMetadata.SCHEMA_VERSION)

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
                properties={_SCHEMA_VERSION_PROPERTY: self._schema_version},
            )
        except pyiceberg.exceptions.TableAlreadyExistsError:
            pass

        self._evolve_schema()
        self._ensure_schema_version_property()

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

    def _ensure_schema_version_property(self) -> None:
        expected = self._schema_version
        for attempt in range(_MAX_RETRIES):
            if self.table.properties.get(_SCHEMA_VERSION_PROPERTY) == expected:
                return
            try:
                self.table.transaction().set_properties(
                    **{_SCHEMA_VERSION_PROPERTY: expected}
                ).commit_transaction()
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
