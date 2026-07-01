from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import pathlib
import typing

import polars
import prefect.runtime.flow_run
import xarray

import data_index.schema
import data_index.schema.metadata


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class ObjectReference(data_index.schema.Schema):
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

    @classmethod
    def to_compressed_base64_table(
        cls,
        object_references: list[typing.Self],
    ) -> str:

        # Set up the schema
        schema = cls.as_polars_schema()

        # Set up df
        if not object_references:
            df = polars.DataFrame(schema=schema)
        else:
            data = [dataclasses.asdict(ref) for ref in object_references]
            df = polars.DataFrame(
                data=data,
                schema=cls.as_polars_schema(),
            )

        # Write to ipc
        buffer = io.BytesIO()

        # Sort to best case for compression
        df = df.sort(
            by=(
                polars.col("bucket"),
                polars.col("version_id"),
                polars.col("key"),
            )
        )

        # Write to buffer
        df.write_ipc(file=buffer, compression="zstd")

        # Base64 encode the compressed bytes
        compressed_base64_table = base64.b64encode(buffer.getvalue()).decode("ascii")

        return compressed_base64_table

    @staticmethod
    def from_compressed_base64_table(base64_str: str) -> list[typing.Self]:
        if not base64_str:
            return []

        # Decode base64 to bytes
        compressed_bytes = base64.b64decode(base64_str)
        buffer = io.BytesIO(compressed_bytes)

        # Polars automatically detects and decompresses the zstd IPC stream
        df = polars.read_ipc(buffer)

        # Reconstruct dataclass instances from the rows
        return [ObjectReference(**row) for row in df.to_dicts()]


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
class DeadLetter(data_index.schema.Schema):
    """
    DeadLetter row schema and backend schema converters.

    `DeadLetter` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    SCHEMA_VERSION: typing.ClassVar[int] = 4
    schema_version: int = SCHEMA_VERSION

    # TODO: Untangle the mess of inheritance and schema for metadata vs dead letter vs object reference
    bucket: str
    key: str
    version_id: str | None
    size: int | None
    hash: str
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
        object_reference: ObjectReference,
        error: str | None,
    ) -> typing.Self:
        return cls(
            bucket=object_reference.bucket,
            key=object_reference.key,
            version_id=object_reference.version_id,
            size=object_reference.size,
            hash=object_reference.hash,
            error=error,
        )

    @classmethod
    def from_object_references(
        cls,
        object_references: list[ObjectReference],
        error: str | None,
    ) -> list[typing.Self]:
        [
            cls.from_object_reference(
                object_reference=object_reference,
                error=error,
            )
            for object_reference in object_references
        ]


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
class MetadataSink(typing.Protocol):
    def provision(self) -> None:
        """Prepare the target store before any writes (e.g. create directories or tables)."""
        ...

    def write(
        self,
        metadata: list[data_index.schema.metadata.StructuredMetadata]
        | list[data_index.schema.metadata.UnstructuredMetadata]
        | list[DeadLetter],
    ) -> None:
        """Persist data"""
        ...
