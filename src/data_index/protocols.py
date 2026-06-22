from __future__ import annotations

import dataclasses
import hashlib
import typing
import urllib.parse

import polars
import xarray

import data_index.structured_metadata


class ObjectReference(typing.NamedTuple):
    bucket: str
    key: str
    version_id: str

    def as_uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def as_versioned_uri(self) -> str:
        version_id = urllib.parse.quote(self.version_id, safe="")
        return f"{self.as_uri()}?versionId={version_id}"

    @classmethod
    def from_s3_uri(
        cls, s3_uri: str, default_version_id: str | None = None
    ) -> ObjectReference:
        parsed = urllib.parse.urlparse(s3_uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid s3 uri: {s3_uri}")

        key = parsed.path.lstrip("/")
        query = urllib.parse.parse_qs(parsed.query)
        version_id = query.get("versionId", [default_version_id])[0]
        if version_id is None:
            raise ValueError(f"Missing versionId in s3 uri: {s3_uri}")

        return cls(bucket=parsed.netloc, key=key, version_id=version_id)

    @property
    def hash(self) -> str:
        """Generates a deterministic 64-character hex string surrogate key."""
        # Use a distinct delimiter to prevent boundary-shifting collisions
        composite = f"bucket:{self.bucket}|key:{self.key}|version:{self.version_id}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()


@dataclasses.dataclass
class BatchEntry:
    object_ref: ObjectReference
    size_bytes: int | None = None

    @property
    def uri(self) -> str:
        """Backward-compatible S3 URI view for components not yet migrated."""
        return self.object_ref.as_uri()


@typing.runtime_checkable
class UnstructuredMetadata(typing.Protocol):
    def load(self) -> dict:
        """Return the full unstructured metadata dict."""
        ...


@dataclasses.dataclass
class RawExtractionResult:
    """Intermediate result returned by MetadataExtractor.extract(). Unstructured metadata
    is a plain dict — persistence wrapping is the responsibility of transform."""

    object_ref: ObjectReference
    structured_metadata: data_index.structured_metadata.StructuredMetadata | None
    unstructured_metadata: dict | None
    status: str  # "succeeded" or "failed"
    error: str | None = None

    @property
    def s3_uri(self) -> str:
        """Backward-compatible S3 URI view for pre-migration consumers."""
        return self.object_ref.as_uri()


@dataclasses.dataclass
class ExtractionResult:
    """Final result returned by _transform_single. Unstructured metadata is a persisted
    UnstructuredMetadata handle (written by metadata_factory during transform)."""

    object_ref: ObjectReference
    structured_metadata: data_index.structured_metadata.StructuredMetadata | None
    unstructured_metadata: UnstructuredMetadata | None
    status: str  # "succeeded" or "failed"
    error: str | None = None

    @property
    def s3_uri(self) -> str:
        """Backward-compatible S3 URI view for pre-migration consumers."""
        return self.object_ref.as_uri()


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
        """Persist Unstructured Metadata dicts keyed by versioned S3 URI."""
        ...


@typing.runtime_checkable
class InventorySource(typing.Protocol):
    def inventory(self) -> polars.DataFrame:
        """Return inventory with required `bucket`,`key`,`version_id`,`size` columns."""
        ...


@typing.runtime_checkable
class BatchPartitioner(typing.Protocol):
    def partition(
        self, inventory: polars.DataFrame
    ) -> typing.Iterator[polars.DataFrame]:
        """Split an inventory DataFrame into a sequence of Batches."""
        ...
