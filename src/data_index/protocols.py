from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import typing

import polars
import prefect.runtime.flow_run
import pyarrow
import xarray

import data_index.schema
import data_index.schema.metadata


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class ObjectReference:
    bucket: str
    key: str
    version_id: str | None
    size: int | None

    def as_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def as_versioned_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}?versionId={self.version_id}"

    def as_path(self) -> pathlib.Path:
        return pathlib.Path(f"{self.bucket}/{self.key}/{self.version_id}")

    @property
    def hash(self) -> str:
        """Generates a deterministic 64-character hex string surrogate key."""
        # Use a distinct delimiter to prevent boundary-shifting collisions
        composite = f"bucket:{self.bucket}|key:{self.key}|version:{self.version_id}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    @property
    def path(self) -> pathlib.Path:
        if self.version_id:
            return pathlib.Path(f"{self.bucket}/{self.key}:{self.version_id}")
        return pathlib.Path(f"{self.bucket}/{self.key}")


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class StagedObject:
    object_reference: ObjectReference
    xarray_handle: XarrayHandle


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class ExtractedObject:
    object_reference: ObjectReference
    extraction_result: ExtractionResult


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class DeadLetter(ObjectReference, data_index.schema.Schema):
    """
    DeadLetter row schema and backend schema converters.

    `DeadLetter` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    SCHEMA_VERSION: typing.ClassVar[int] = 1
    schema_version: int = SCHEMA_VERSION

    error: str | None
    index_flow_id: str | None = dataclasses.field(
        default_factory=lambda: prefect.runtime.flow_run.get_parent_flow_run_id()
    )
    batch_flow_id: str | None = dataclasses.field(
        default_factory=lambda: prefect.runtime.flow_run.get_id()
    )

    @classmethod
    def from_object_reference(
        cls,
        object_reference: data_index.protocols.ObjectReference,
        error: str | None,
    ) -> typing.Self:
        return cls(
            bucket=object_reference.bucket,
            key=object_reference.key,
            version_id=object_reference.version_id,
            size=object_reference.size,
            error=error,
        )


@dataclasses.dataclass
class ExtractionResult:
    """Final result returned by _transform_single. Unstructured metadata is a persisted
    UnstructuredMetadata handle (written by metadata_factory during transform)."""

    structured_metadata: data_index.schema.metadata.StructuredMetadata | None
    unstructured_metadata: data_index.schema.metadata.UnstructuredMetadata | None


@typing.runtime_checkable
class XarrayHandle(typing.Protocol):
    object_ref: ObjectReference
    file_format: str | None

    @property
    def s3_uri(self) -> str: ...

    @property
    def ds(self) -> xarray.Dataset:
        """Return a handle-local singleton xarray dataset."""
        ...

    def cleanup(self) -> None:
        """Release any resources associated with this handle (e.g. delete a local file)."""
        ...


@typing.runtime_checkable
class InventorySource(typing.Protocol):
    def inventory(self) -> polars.DataFrame:
        """Return inventory with required `bucket`,`key`,`version_id`,`size` columns."""
        ...


@typing.runtime_checkable
class BatchPartitioner(typing.Protocol):
    def partition(
        self,
        inventory: polars.DataFrame,
    ) -> typing.Iterator[list[ObjectReference]]:
        """Split an inventory DataFrame into a sequence of Batches."""
        ...


@typing.runtime_checkable
class FileFetcher(typing.Protocol):
    def fetch(
        self, object_references: list[ObjectReference]
    ) -> tuple[list[StagedObject], list[DeadLetter]]:
        """Instantiate a list of XarrayHandle to be consumed by a Metadata Extractor."""
        ...


@typing.runtime_checkable
class MetadataExtractor(typing.Protocol):
    def extract(self, staged_object: StagedObject) -> ExtractedObject | DeadLetter:
        """Extract structured and unstructured metadata from an XarrayHandle."""
        ...


@typing.runtime_checkable
class Sink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self,
        data: pyarrow.Table,
    ) -> list[DeadLetter]:
        """Persist data"""
        ...


@typing.runtime_checkable
class StructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self,
        data: list[data_index.schema.metadata.StructuredMetadata],
    ) -> list[DeadLetter]:
        """Persist Structured Metadata"""
        ...


@typing.runtime_checkable
class UnstructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self, data: list[data_index.schema.metadata.UnstructuredMetadata]
    ) -> list[DeadLetter]:
        """Persist Unstructured Metadata"""
        ...
