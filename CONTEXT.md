# Data Index

A pipeline that ingests CF-compliant NetCDF files from S3, extracts metadata, and stores it for discovery and later analysis.

## Language

**Batch**:
A caller-supplied manifest of S3 URIs (with file sizes) representing the set of NetCDF files to process in one pipeline run. All URIs in a batch must be unique. The total size must not exceed the configured batch size limit. File sizes are used by `FileFetcher` to route each file to the appropriate fetch strategy.
_Avoid_: job, dataset, collection

**Structured Metadata**:
A fixed-schema set of fields extracted from a NetCDF file — including spatial extent, temporal range, CRS, and `s3_uri`. Stored as a Polars DataFrame and persisted as Parquet (target: S3 Table).
_Avoid_: metadata (without qualifier)

**Unstructured Metadata**:
The full set of global attributes, variable metadata, and coordinate metadata from a NetCDF file. Represented by the `UnstructuredMetadata` protocol. The default implementation (`DiskCachedUnstructuredMetadata`) writes to diskcache immediately on construction and reads back on `load()`. Schema varies by dataset.
_Avoid_: raw metadata, attributes

**Extraction Result**:
The dataclass returned by `_transform_single` for a single file. Contains: one structured metadata row, one `UnstructuredMetadata` instance (or `None` on failure), and a status (succeeded/failed with error message). Passed directly to `load`.

## Relationships

- A **Batch** is consumed by `extract` to produce a list of **XarrayHandle** objects
- A list of **XarrayHandle** objects is consumed by `transform`; one `_transform_single` task runs per handle
- Each `_transform_single` produces one **Extraction Result**; unstructured metadata is persisted immediately by calling the injected `metadata_factory(s3_uri, data)`
- **Extraction Results** are returned by `transform` and consumed by `load`
- `load` delegates persistence to an injected **StructuredSink** and **UnstructuredSink**
- `extract` delegates file fetching to an injected **FileFetcher**
- `transform` delegates metadata extraction to an injected **MetadataExtractor** and unstructured persistence to an injected `metadata_factory` callable

## Plugin Protocols

**XarrayHandle**:
A lazy reference to a NetCDF dataset, carrying its `s3_uri` and resolving to an open `xarray.Dataset` on demand. The handoff between `extract` and `transform`. Provides `cleanup()` to release resources after use (no-op for cloud handles; deletes the local file for disk handles). Two implementations exist: `S3XarrayHandle` (cloud-native targeted byte reads via fsspec, optimal for large files) and `DiskXarrayHandle` (full download to local disk, optimal for small files). The extractor does not need to know which strategy was used.
_Avoid_: handle, file handle, dataset handle

**FileFetcher**:
A pluggable component that, given a validated list of S3 URIs, returns a list of `XarrayHandle` objects. The planned production implementation (`ThresholdFileFetcher`) routes each file to a `DiskXarrayHandle` (full download) or `S3XarrayHandle` (cloud-native byte reads) based on a configurable size threshold.
_Avoid_: downloader, fetcher, sync class

**MetadataExtractor**:
A pluggable component that, given an open `xarray.Dataset` and its `s3_uri`, extracts both Structured and Unstructured Metadata in a single pass and returns a `RawExtractionResult` (with unstructured metadata as a plain `dict`). The `transform` step is responsible for persisting that dict via `metadata_factory`.
_Avoid_: parser, reader

**StructuredSink**:
A pluggable component that persists a Structured Metadata DataFrame to a target store (e.g. Parquet, S3 Table).
_Avoid_: writer, exporter

**UnstructuredSink**:
A pluggable component that persists Unstructured Metadata dicts (keyed by `s3_uri`) to a final destination store (e.g. Parquet, DynamoDB). Receives `dict[str, dict]` — callers resolve `UnstructuredMetadata.load()` before passing.
_Avoid_: writer, exporter

## Constraints

- `XarrayHandle.cleanup()` is called after `transform` completes; implementations decide what cleanup means (e.g. `DiskXarrayHandle` deletes the local file, `S3XarrayHandle` is a no-op)
- The caller is responsible for ensuring no `s3_uri` appears more than once across pipeline runs
- diskcache is keyed by `s3_uri`
- The `StructuredSink` enforces the Structured Metadata schema on write

## Example dialogue

> **Dev:** "Should we re-index a file if we've already seen it?"
> **Domain expert:** "The upstream process controls the **Batch** — it shouldn't send us the same `s3_uri` twice. We trust it to handle that."

> **Dev:** "Why do we store **Unstructured Metadata** in diskcache instead of just a JSON file?"
> **Domain expert:** "The schema varies wildly across datasets. We need concurrent writes from Prefect tasks and a format we can query later. Diskcache gives us that before we decide on a permanent store like DynamoDB."

## Flagged ambiguities

- "metadata" (unqualified) was used to mean both **Structured Metadata** and **Unstructured Metadata** — resolved: always qualify the term.
