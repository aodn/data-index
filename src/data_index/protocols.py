from __future__ import annotations

import dataclasses
import typing

import polars
import xarray


@dataclasses.dataclass
class BatchEntry:
    uri: str
    size_bytes: int | None = None


@dataclasses.dataclass
class StructuredMetadata:
    s3_uri: str
    lat_min: float | None
    lat_max: float | None
    lon_min: float | None
    lon_max: float | None
    time_min: str | None
    time_max: str | None
    crs: str | None
    file_format: str | None = None
    collection: str | None = None

    polars_schema: typing.ClassVar[polars.Schema] = polars.Schema({
        "s3_uri": polars.String,
        "lat_min": polars.Float64,
        "lat_max": polars.Float64,
        "lon_min": polars.Float64,
        "lon_max": polars.Float64,
        "time_min": polars.String,
        "time_max": polars.String,
        "crs": polars.String,
        "file_format": polars.String,
        "collection": polars.String,
    })


class UnstructuredMetadata(typing.Protocol):
    def load(self) -> dict:
        """Return the full unstructured metadata dict."""
        ...


@dataclasses.dataclass
class RawExtractionResult:
    """Intermediate result returned by MetadataExtractor.extract(). Unstructured metadata
    is a plain dict — persistence wrapping is the responsibility of transform."""
    s3_uri: str
    structured_metadata: StructuredMetadata | None
    unstructured_metadata: dict | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


@dataclasses.dataclass
class ExtractionResult:
    """Final result returned by _transform_single. Unstructured metadata is a persisted
    UnstructuredMetadata handle (written by metadata_factory during transform)."""
    s3_uri: str
    structured_metadata: StructuredMetadata | None
    unstructured_metadata: UnstructuredMetadata | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


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

class FileFetcher(typing.Protocol):
    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:
        """Instantiate a list of XarrayHandle to be consumed by a Metadata Extractor."""
        ...


class MetadataExtractor(typing.Protocol):
    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        """Extract structured and unstructured metadata from an XarrayHandle."""
        ...


class StructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(self, data: list[StructuredMetadata]) -> None:
        """Persist Structured Metadata to a target store."""
        ...


class UnstructuredSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(self, data: dict[str, dict]) -> None:
        """Persist Unstructured Metadata dicts (keyed by s3_uri) to a target store."""
        ...


class InventorySource(typing.Protocol):
    def inventory(self) -> polars.DataFrame:
        """Return the full corpus inventory as a DataFrame with `s3_uri` and `size` columns."""
        ...


class BatchPartitioner(typing.Protocol):
    def partition(self, inventory: polars.DataFrame) -> typing.Iterator[polars.DataFrame]:
        """Split an inventory DataFrame into a sequence of Batches."""
        ...
