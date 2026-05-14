from __future__ import annotations

import dataclasses
import pathlib
import typing

import polars
import xarray
import pydantic


class ManifestEntry(pydantic.BaseModel):
    s3_uri: str
    absolute_path: pathlib.Path


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
class ExtractionResult:
    s3_uri: str
    structured_metadata: StructuredMetadata | None
    unstructured_metadata: UnstructuredMetadata | None
    status: str  # "succeeded" or "failed"
    error: str | None = None


class FileFetcher(typing.Protocol):
    def fetch(self, uris: list[str], extract_path: pathlib.Path) -> list[ManifestEntry]:
        """Download files and return a Manifest as a list of ManifestEntry."""
        ...


class MetadataExtractor(typing.Protocol):
    def extract(self, ds: xarray.Dataset, s3_uri: str) -> ExtractionResult:
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
