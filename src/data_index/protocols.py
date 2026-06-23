from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import typing

import polars
import xarray

import data_index.schema.metadata


class ObjectReference(typing.NamedTuple):
    bucket: str
    key: str
    version_id: str | None
    size: int | None
    xarray_handle: XarrayHandle | None

    def as_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def as_versioned_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}?versionId={self.version_id}"

    def as_path(self) -> pathlib.Path:
        return pathlib.Path(f"{self.bucket}/{self.key}/{self.version_id}")

    def with_xarray_handle(self, xarray_handle: XarrayHandle) -> ObjectReference:
        """
        Returns a new ObjectReference with the updated xarray_handle.

        Note that NamedTuple._replace is actually a public method despite having
        an underscore.
        """
        return self._replace(xarray_handle=xarray_handle)

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


@dataclasses.dataclass
class ExtractionResult:
    """Final result returned by _transform_single. Unstructured metadata is a persisted
    UnstructuredMetadata handle (written by metadata_factory during transform)."""

    structured_metadata: data_index.schema.metadata.StructuredMetadata | None
    unstructured_metadata: data_index.schema.metadata.UnstructuredMetadata | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


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
    def fetch(self, object_references: list[ObjectReference]) -> list[ObjectReference]:
        """Instantiate a list of XarrayHandle to be consumed by a Metadata Extractor."""
        ...


@typing.runtime_checkable
class MetadataExtractor(typing.Protocol):
    def extract(self, object_reference: ObjectReference) -> ExtractionResult:
        """Extract structured and unstructured metadata from an XarrayHandle."""
        ...


@typing.runtime_checkable
class StructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(self, data: list[data_index.schema.metadata.StructuredMetadata]) -> None:
        """Persist Structured Metadata"""
        ...


@typing.runtime_checkable
class UnstructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self, data: list[data_index.schema.metadata.UnstructuredMetadata]
    ) -> None:
        """Persist Unstructured Metadata"""
        ...
