# Data Index

A pipeline that ingests CF-compliant NetCDF files from S3, extracts metadata, and stores it for discovery and later analysis.

## Language

**Batch**:
A caller-supplied manifest of S3 URIs (with file sizes) representing the set of NetCDF files to process in one pipeline run. All URIs in a batch must be unique. The total size must not exceed the configured batch size limit. Represented as a Polars DataFrame with columns `s3_uri` and `size`.
_Avoid_: job, dataset, collection

**Batch Entry**:
A single item within a **Batch** — an `s3_uri` paired with an optional `size_bytes`. Passed to `FileFetcher.fetch()` so routing decisions (e.g. size threshold) can be made before any I/O.
_Avoid_: row, record, file

**Structured Metadata**:
A fixed-schema set of fields extracted from a NetCDF file — including spatial extent, temporal range, CRS, file format, `collection`, and `s3_uri`. Stored as a Polars DataFrame and persisted to an S3 Table (Apache Iceberg). `collection` is the second path segment of the S3 key (e.g. `ANMN` from `IMOS/ANMN/NSW/file.nc`), populated by the **MetadataExtractor**. The current extractor derives extent from coordinate arrays; a future v2 extractor will use ACDD global attributes where available (see ADR-0005).
_Avoid_: metadata (without qualifier)

**Unstructured Metadata**:
The full set of global attributes, variable metadata, coordinate metadata, and file format from a NetCDF file. Represented by the `UnstructuredMetadata` protocol. The default implementation (`DiskCachedUnstructuredMetadata`) writes to diskcache immediately on construction and reads back on `load()`. Schema varies by dataset.
_Avoid_: raw metadata, attributes

**Extraction Result**:
The dataclass returned by `_transform_single` for a single file. Contains: one structured metadata row, one `UnstructuredMetadata` instance (or `None` on failure), and a status (succeeded/failed with error message). Passed directly to `load`.

**File Format**:
The NetCDF encoding of a file — one of `NETCDF3_CLASSIC`, `NETCDF3_64BIT_OFFSET`, `NETCDF5`, or `NETCDF4` (HDF5). Determined by reading the first 8 magic bytes of the file via the `XarrayHandle`. Captured in both **Structured Metadata** and **Unstructured Metadata** to inform future routing and extraction strategy decisions. The IMOS corpus is a mixture of formats.
_Avoid_: version, encoding, NetCDF version

## Relationships

- An **InventorySource** provides the inventory DataFrame consumed by a **BatchPartitioner**
- A **BatchPartitioner** produces a sequence of **Batches** consumed by the **Orchestrator**
- The **Orchestrator** dispatches each **Batch** to a Fargate worker task that runs the full extract → transform → load pipeline
- The **Orchestrator** reads and writes the **Run State File** to track which Batches have been processed
- A **Batch** is consumed by `extract` to produce a list of **XarrayHandle** objects
- A list of **XarrayHandle** objects is consumed by `transform`; one `_transform_single` task runs per handle
- Each `_transform_single` passes the full **XarrayHandle** to **MetadataExtractor** (not just the dataset), then produces one **Extraction Result**; unstructured metadata is persisted immediately by calling the injected `metadata_factory(s3_uri, data)`
- **Extraction Results** are returned by `transform` and consumed by `load`
- `load` delegates persistence to an injected **StructuredSink** and **UnstructuredSink**
- `extract` delegates file fetching to an injected **FileFetcher**, passing a list of **Batch Entry** objects
- `transform` delegates metadata extraction to an injected **MetadataExtractor** and unstructured persistence to an injected `metadata_factory` callable

## Cluster Run

**Orchestrator**:
The Prefect flow that reads an **InventorySource**, partitions it into **Batches** via a **BatchPartitioner**, consults the **Run State File** to skip already-completed Batches, and dispatches remaining Batches as tasks to a Fargate Dask cluster. Calls `provision()` on both injected sinks before any Batch is dispatched.
_Avoid_: coordinator, driver, master, controller

**InventorySource**:
A pluggable component that provides the full corpus inventory as a DataFrame with `s3_uri` and `size` columns. Implementations: `ParquetInventorySource` (reads a local or S3 Parquet file) and `LiveS3InventorySource` (queries an S3 inventory table at runtime).
_Avoid_: file list, manifest, catalogue

**BatchPartitioner**:
A pluggable component that splits an inventory DataFrame into a sequence of **Batches**, each satisfying configured file-count and size limits. Implementations: `GreedyBatchPartitioner` (pure size-based bin-packing) and `CollectionGroupedBatchPartitioner` (groups files by S3 prefix/collection before bin-packing).
_Avoid_: chunker, splitter, batcher

**Run State File**:
An S3 JSON file written and read by the **Orchestrator** to track which Batch IDs have completed successfully. Enables a run to resume from the point of failure without re-processing completed Batches. Batch ID is a deterministic hash of the Batch's sorted `s3_uri` list.
_Avoid_: checkpoint, progress file, resume file

## Plugin Protocols

**XarrayHandle**:
A lazy reference to a NetCDF dataset, carrying its `s3_uri`, `file_format` (from magic bytes), and resolving to an open `xarray.Dataset` on demand. The handoff between `extract` and `transform`. Provides `cleanup()` to release resources after use (no-op for cloud handles; deletes the local file for disk handles). Two implementations exist: `S3XarrayHandle` (cloud-native targeted byte reads via fsspec, optimal for large files) and `DiskXarrayHandle` (full download to local disk, optimal for small files). The extractor does not need to know which strategy was used.
_Avoid_: handle, file handle, dataset handle

**FileFetcher**:
A pluggable component that, given a list of **Batch Entry** objects, returns a list of `XarrayHandle` objects. The production implementation (`ThresholdFileFetcher`) routes each file to a `DiskXarrayHandle` (full download) or `S3XarrayHandle` (cloud-native byte reads) based on a configurable size threshold. Entries with no size default to the cloud path. The disk implementation (`S5CMDFetcher`) wraps s5cmd's `run` command, which defaults to 256 internal parallel workers — appropriate for an isolated Fargate container but must be capped for local runs shared with other processes.
_Avoid_: downloader, fetcher, sync class

**MetadataExtractor**:
A pluggable component that, given an **XarrayHandle**, extracts both Structured and Unstructured Metadata in a single pass and returns a `RawExtractionResult` (with unstructured metadata as a plain `dict`). Receives the handle — not a raw `xarray.Dataset` — so it can read `file_format` from magic bytes without reopening the file. The `transform` step is responsible for persisting the dict via `metadata_factory`.
_Avoid_: parser, reader

**StructuredSink**:
A pluggable component that prepares and persists a Structured Metadata DataFrame to a target store. All implementations expose `provision()` — called once by the **Orchestrator** before Batches are dispatched — and `write()` for each Batch. The production implementation (`StructuredS3TableSink`) writes to an S3 Table (Apache Iceberg) via PyIceberg, partitioned by `collection` then year (from `time_min`; null `time_min` goes to the null partition bucket); appends on each write and retries on OCC conflicts. The local implementation (`StructuredParquetSink`) creates the output directory on `provision()` and writes a Parquet file on each `write()`.
_Avoid_: writer, exporter

**UnstructuredSink**:
A pluggable component that prepares and persists Unstructured Metadata dicts (keyed by `s3_uri`) to a final destination store. Receives `dict[str, dict]` — callers resolve `UnstructuredMetadata.load()` before passing. All implementations expose `provision()` and `write()`. The production implementation (`UnstructuredS3TableSink`) writes to an S3 Table (Apache Iceberg) via PyIceberg with schema `(s3_uri STRING, collection STRING, metadata STRING)` where `collection` is derived from the `s3_uri` key at write time and `metadata` is JSON-encoded.
_Avoid_: writer, exporter

## Constraints

- `XarrayHandle.cleanup()` is called after `transform` completes; implementations decide what cleanup means (e.g. `DiskXarrayHandle` deletes the local file, `S3XarrayHandle` is a no-op)
- `sink.provision()` must be called before any `sink.write()` calls — the **Orchestrator**'s `pre_run` hook is the canonical place to do this
- The caller is responsible for ensuring no `s3_uri` appears more than once across pipeline runs
- diskcache is keyed by `s3_uri`
- The `StructuredSink` enforces the Structured Metadata schema on write
- **File Format** is determined from magic bytes (first 8 bytes of the file), not from xarray internals — avoids coupling to private xarray/netCDF4 attributes
- NetCDF4/HDF5 files require multiple block fetches to traverse the HDF5 btree even for metadata-only reads; this makes `S3XarrayHandle` slower for large NetCDF4 files than for NetCDF3
- **Three-layer concurrency model**: peak resource usage is `batch_workers × transform_threads × s5cmd_workers`. On Fargate each worker runs in an isolated container so all three can be maximised independently. For local runs all three layers share the same machine and must be capped together to avoid memory exhaustion and CPU saturation.

## Example dialogue

> **Dev:** "Should we re-index a file if we've already seen it?"
> **Domain expert:** "The upstream process controls the **Batch** — it shouldn't send us the same `s3_uri` twice. We trust it to handle that."

> **Dev:** "Why do we store **Unstructured Metadata** in diskcache instead of just a JSON file?"
> **Domain expert:** "The schema varies wildly across datasets. We need concurrent writes from Prefect tasks and a format we can query later. Diskcache gives us that before we decide on a permanent store like DynamoDB."

> **Dev:** "Why doesn't the extractor use `geospatial_lat_min` from global attributes instead of computing min/max from the coordinate arrays?"
> **Domain expert:** "We don't know yet which files carry those ACDD attributes reliably. The first pipeline run will collect **Unstructured Metadata** for every file — we'll analyse the attribute distribution and build a v2 extractor once we know what's actually there."

> **Dev:** "Why does `ThresholdFileFetcher` default unknown file sizes to the cloud path?"
> **Domain expert:** "If we don't know the size, downloading could be arbitrarily expensive. Cloud byte-range reads are safe for any size — they just cost more per-request for very small files."

## Flagged ambiguities

- "metadata" (unqualified) was used to mean both **Structured Metadata** and **Unstructured Metadata** — resolved: always qualify the term.
- "file format" was initially derived from xarray private internals (`_file_obj._ds.file_format`) — resolved: always read from magic bytes via `XarrayHandle.file_format`.
