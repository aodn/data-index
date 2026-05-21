from __future__ import annotations

import dataclasses
import typing

import polars
import xarray


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

    polars_schema: typing.ClassVar[polars.Schema] = polars.Schema({
        "s3_uri": polars.String,
        "lat_min": polars.Float64,
        "lat_max": polars.Float64,
        "lon_min": polars.Float64,
        "lon_max": polars.Float64,
        "time_min": polars.String,
        "time_max": polars.String,
        "crs": polars.String,
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

    @property
    def ds(self) -> xarray.Dataset:
        """Return an xarray dataset"""
        ...

    def cleanup(self) -> None:
        """Release any resources associated with this handle (e.g. delete a local file)."""
        ...

class FileFetcher(typing.Protocol):
    def fetch(self, uris: list[str]) -> list[XarrayHandle]:
        """Instantiate a list of XarrayHandle to be consumed by a Metadata Extractor."""
        ...


class MetadataExtractor(typing.Protocol):
    def extract(self, ds: xarray.Dataset, s3_uri: str) -> RawExtractionResult:
        """Extract structured and unstructured metadata from an open xarray Dataset."""
        ...


class StructuredSink(typing.Protocol):
    def write(self, data: list[StructuredMetadata]) -> None:
        """Persist Structured Metadata to a target store."""
        ...


class UnstructuredSink(typing.Protocol):
    def write(self, data: dict[str, dict]) -> None:
        """Persist Unstructured Metadata dicts (keyed by s3_uri) to a target store."""
        ...
