from __future__ import annotations

import dataclasses
import typing

import polars
import xarray

import data_index.structured_metadata


@dataclasses.dataclass
class BatchEntry:
    uri: str
    size_bytes: int | None = None


@typing.runtime_checkable
class UnstructuredMetadata(typing.Protocol):
    def load(self) -> dict:
        """Return the full unstructured metadata dict."""
        ...


@dataclasses.dataclass
class RawExtractionResult:
    """Intermediate result returned by MetadataExtractor.extract(). Unstructured metadata
    is a plain dict — persistence wrapping is the responsibility of transform."""

    s3_uri: str
    structured_metadata: data_index.structured_metadata.StructuredMetadata | None
    unstructured_metadata: dict | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


@dataclasses.dataclass
class ExtractionResult:
    """Final result returned by _transform_single. Unstructured metadata is a persisted
    UnstructuredMetadata handle (written by metadata_factory during transform)."""

    s3_uri: str
    structured_metadata: data_index.structured_metadata.StructuredMetadata | None
    unstructured_metadata: UnstructuredMetadata | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


@typing.runtime_checkable
class XarrayHandle(typing.Protocol):
    s3_uri: str
    file_format: str | None

    @property
    def ds(self) -> xarray.Dataset:
        """Return an xarray dataset"""
        ...

    def cleanup(self) -> None:
        """Release any resources associated with this handle (e.g. delete a local file)."""
        ...


@typing.runtime_checkable
class FileFetcher(typing.Protocol):
    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:
        """Instantiate a list of XarrayHandle to be consumed by a Metadata Extractor."""
        ...


@typing.runtime_checkable
class MetadataExtractor(typing.Protocol):
    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        """Extract structured and unstructured metadata from an XarrayHandle."""
        ...


@typing.runtime_checkable
class StructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self, data: list[data_index.structured_metadata.StructuredMetadata]
    ) -> None:
        """Persist Structured Metadata to a target store."""
        ...


@typing.runtime_checkable
class UnstructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(self, data: dict[str, dict]) -> None:
        """Persist Unstructured Metadata dicts (keyed by s3_uri) to a target store."""
        ...


@typing.runtime_checkable
class InventorySource(typing.Protocol):
    def inventory(self) -> polars.DataFrame:
        """Return the full corpus inventory as a DataFrame with `s3_uri` and `size` columns."""
        ...


@typing.runtime_checkable
class BatchPartitioner(typing.Protocol):
    def partition(
        self, inventory: polars.DataFrame
    ) -> typing.Iterator[polars.DataFrame]:
        """Split an inventory DataFrame into a sequence of Batches."""
        ...
