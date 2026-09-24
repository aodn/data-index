# Data Index

A pipeline that ingests CF-compliant NetCDF files from S3, extracts metadata, and stores it for discovery and later analysis.

## Language

**Batch**:
A caller-supplied manifest of object versions (with file sizes) representing the set of NetCDF files to process in one pipeline run. All **Object Version Identity** values in a batch must be unique. The total size must not exceed the configured batch size limit. Represented as a Polars DataFrame with required columns `bucket`, `key`, `version_id`, and `size` (extra columns allowed).
_Avoid_: job, dataset, collection

**Batch Entry**:
A single item within a **Batch** — (`bucket`, `key`, `version_id`) paired with an optional `size_bytes`. Passed to `FileFetcher.fetch()` so routing decisions (e.g. size threshold) can be made before any I/O.
_Avoid_: row, record, file

**Object Version Identity**:
The canonical identity tuple for a source object: (`bucket`, `key`, `version_id`), where all three fields are required and non-null. For S3 sources, `version_id` is the real S3 object version. For local filesystem sources, `version_id` is the sentinel `__LOCAL__`; `bucket` is the absolute source path prefix up to and including the configured bucket anchor segment; and `key` is the path suffix after that anchor.
_Avoid_: object key, uri-only identity, `s3_uri`-only identity

**Object Reference**:
A lightweight immutable value object (named tuple) that carries (`bucket`, `key`, `version_id`) across protocol boundaries, and exposes derived convenience properties for `hash` and `facility`.
_Avoid_: parsing identity from strings repeatedly

**Object Reference Hash**:
A deterministic SHA-256 hash derived from **Object Version Identity**, used as sink join/upsert key while preserving explicit identity fields.
_Avoid_: hash-only identity

**Facility**:
The second path segment of the S3 key under `IMOS/` (for example, `ANMN` from `IMOS/ANMN/NSW/file.nc`). This is the canonical domain term for top-level IMOS grouping; when not derivable, use sentinel `UNKNOWN`.
_Avoid_: collection

**Structured Metadata**:
A fixed-schema set of fields extracted from a NetCDF file — including **Object Version Identity** (`bucket`, `key`, `version_id`), spatial extent, temporal range, CRS, file format, and `facility`. Stored as a Polars DataFrame and persisted to an S3 Table (Apache Iceberg). `facility` is populated by the **MetadataExtractor** from the second path segment. The current extractor derives extent from coordinate arrays; a future v2 extractor will use ACDD global attributes where available (see ADR-0005).
_Avoid_: metadata (without qualifier)

**Unstructured Metadata**:
The full set of global attributes, variable metadata, coordinate metadata, and file format from a NetCDF file, represented as an `UnstructuredMetadata` row contract with fields (`bucket`, `key`, `version_id`, `hash`, `schema_version`, `file_format`, `facility`, `metadata`) where `metadata` is a dict payload.
_Avoid_: raw metadata, attributes

**Extraction Result**:
The dataclass returned by `_transform_single` for a single file. Contains: one `object_ref`, one structured metadata row, one `UnstructuredMetadata` instance (or `None` on failure), and a status (succeeded/failed with error message). Passed directly to `load`.

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
- Each `_transform_single` passes the full **XarrayHandle** to **MetadataExtractor** (not just the dataset), then builds one **Extraction Result** with final `StructuredMetadata` and `UnstructuredMetadata` rows
- **Extraction Results** are returned by `transform` and consumed by `load`
- `load` delegates persistence to an injected **StructuredSink** and **UnstructuredSink**
- `extract` delegates file fetching to an injected **FileFetcher**, passing a list of **Batch Entry** objects
- `transform` delegates metadata extraction to an injected **MetadataExtractor** and unstructured persistence to an injected `metadata_factory` callable

## Cluster Run

**Orchestrator**:
The Prefect flow that reads an **InventorySource**, partitions it into **Batches** via a **BatchPartitioner**, consults the **Run State File** to skip already-completed Batches, and dispatches remaining Batches as tasks to a Fargate Dask cluster. Calls `provision()` on both injected sinks before any Batch is dispatched.
_Avoid_: coordinator, driver, master, controller

**InventorySource**:
A pluggable component that provides the full corpus inventory as a DataFrame with required columns `bucket`, `key`, `version_id`, and `size` (extra columns allowed). For local filesystem sources, inventory discovery is path-based and uses a configured `root_path` plus a glob pattern relative to that root, with an optional deterministic `max_files` cap applied after sorting by `key`.
_Avoid_: file list, manifest, catalogue

**BatchPartitioner**:
A pluggable component that splits an inventory DataFrame into a sequence of **Batches**, each satisfying configured file-count and size limits. Implementations: `GreedyBatchPartitioner` (pure size-based bin-packing) and `FacilityGroupedBatchPartitioner` (groups files by S3 prefix/facility before bin-packing).
_Avoid_: chunker, splitter, batcher

**Run State File**:
An S3 JSON file written and read by the **Orchestrator** to track which Batch IDs have completed successfully. Enables a run to resume from the point of failure without re-processing completed Batches. Batch ID is a deterministic hash of the Batch's sorted (`bucket`, `key`, `version_id`) identities.
_Avoid_: checkpoint, progress file, resume file

## Plugin Protocols

**XarrayHandle**:
A lazy reference to a NetCDF dataset, carrying its `object_ref`, `file_format` (from magic bytes), and resolving to an open `xarray.Dataset` on demand. The handoff between `extract` and `transform`. Provides `cleanup()` to release resources after use (no-op for cloud handles; deletes the local file for disk handles). Two implementations exist: `S3XarrayHandle` (cloud-native targeted byte reads via fsspec, optimal for large files) and `DiskXarrayHandle` (full download to local disk, optimal for small files). The extractor does not need to know which strategy was used.
_Avoid_: handle, file handle, dataset handle

**FileFetcher**:
A pluggable component that, given a list of **Batch Entry** objects, returns a list of `XarrayHandle` objects. The production implementation (`ThresholdFileFetcher`) routes each file to a `DiskXarrayHandle` (full download) or `S3XarrayHandle` (cloud-native byte reads) based on a configurable size threshold. Entries with no size default to the cloud path. Fetching must be pinned to the requested `version_id` (not latest object). The disk implementation (`S5CMDFetcher`) wraps s5cmd's `run` command, which defaults to 256 internal parallel workers — appropriate for an isolated Fargate container but must be capped for local runs shared with other processes.
_Avoid_: downloader, fetcher, sync class

**MetadataExtractor**:
A pluggable component that, given an **XarrayHandle**, extracts both Structured and Unstructured Metadata in a single pass and returns a `RawExtractionResult` (with unstructured metadata serialized as a JSON string payload). Receives the handle — not a raw `xarray.Dataset` — so it can read `file_format` from magic bytes without reopening the file.
_Avoid_: parser, reader

**StructuredSink**:
A pluggable component that prepares and persists Structured Metadata rows to a target store. All implementations expose `provision()` — called once by the **Orchestrator** before Batches are dispatched — and `write()` for each Batch. The default implementation (`IcebergTableSink`) writes to an Iceberg table via PyIceberg with hash-key upserts and retry-on-conflict behavior. Environment is chosen by catalog config: S3 Tables REST catalog in production, SQLite catalog + local warehouse for local runs. An opt-in `DynamoDBSink` implementation is also available for structured rows and uses `hash` as the item key.
_Avoid_: writer, exporter

**UnstructuredSink**:
A pluggable component that prepares and persists Unstructured Metadata rows to a final destination store. Receives `UnstructuredMetadata` rows with explicit identity fields, required `facility`, top-level `file_format`, and a `metadata` JSON-string payload. All implementations expose `provision()` and `write()`. The default implementation (`IcebergTableSink`) writes to Iceberg with hash-key upserts (latest write wins), with backend selected by catalog config (S3 Tables in production, SQLite + local warehouse for local runs). An opt-in `DynamoDBSink` implementation is available for unstructured rows, also keyed by `hash` with latest-write-wins behavior.
_Avoid_: writer, exporter

## Constraints

- `XarrayHandle.cleanup()` is called after `transform` completes; implementations decide what cleanup means (e.g. `DiskXarrayHandle` deletes the local file, `S3XarrayHandle` is a no-op)
- `sink.provision()` must be called before any `sink.write()` calls — the **Orchestrator**'s `pre_run` hook is the canonical place to do this
- Schema-breaking sink transitions use an explicit opt-in reset mode in `provision()` (drop/recreate), not implicit destructive behavior
- Sink implementations own idempotency for **Object Version Identity** by upserting on `hash` (latest write wins per object version); this applies forward from the upsert change and does not retroactively deduplicate historical append-era duplicates
- DynamoDB sinks use `hash` as the item key; structured and unstructured writes must target separate tables to avoid cross-type overwrites
- DynamoDB base table writes use `hash` as partition key for ingest distribution; facility-based access paths are modeled through GSIs, not the base key
- DynamoDB facility replay access path uses a GSI with `facility` partition key and `indexed_at_ms` sort key for incremental, checkpointed draining
- DynamoDB facility GSIs project keys only to reduce write amplification; drain workers fetch full rows from base tables by key
- DynamoDB replay semantics are at-least-once at checkpoint boundaries; downstream sinks rely on hash idempotency
- Sink upsert/join operations use **Object Reference Hash**; `bucket`, `key`, and `version_id` remain required identity fields in all pipeline contracts and sink schemas
- **Object Reference Hash** generation (`sha256(bucket/key/version composite)`) is a stable contract; changes require explicit schema/version migration
- Identity fields (`bucket`, `key`, `version_id`) are required and non-null at pipeline boundaries
- Pipeline stages fail fast if any identity field (`bucket`, `key`, `version_id`) is null/empty
- Structured and Unstructured sink contracts stay consistent on identity model and upsert key semantics
- Missing/invalid facility derivations are coerced to sentinel `UNKNOWN` (not null)
- File fetch/read operations are pinned to `version_id`; extraction must read the exact requested source identity (`S3 version_id` for S3, `__LOCAL__` sentinel for local files)
- Local filesystem fetchers must reject non-`__LOCAL__` version IDs as invalid identity for local-path fetching
- Local filesystem inventory sources must emit `version_id = __LOCAL__`
- Local filesystem inventory sources expose `local_version_id` config with default `__LOCAL__`
- Local filesystem inventory sources must emit `bucket` as the absolute path prefix up to the configured bucket anchor segment (first occurrence in the resolved path)
- Local filesystem inventory sources must emit `key` as the suffix after the emitted `bucket` prefix
- Local filesystem inventory sources must fail fast if a discovered path does not contain the configured bucket anchor segment
- Local filesystem inventory sources must stat each discovered file and emit byte `size`
- Local filesystem inventory sources must return deterministic order (sorted by `key`)
- Local filesystem inventory sources may apply an optional deterministic `max_files` cap after sorting by `key`
- Local filesystem inventory sources must deduplicate canonical (`bucket`, `key`, `version_id`) identities before returning
- Local filesystem inventory sources must fail fast on invalid `root_path` configuration (missing/non-directory)
- Local filesystem inventory sources must fail fast if discovered paths become unreadable during scan/stat
- Local filesystem inventory sources must reject symlink-resolved paths that escape `root_path`
- Local filesystem inventory sources include only regular files; non-file glob matches are skipped
- Local filesystem fetchers must resolve one candidate file path as `bucket / key` and require it to exist as a regular file
- Local filesystem fetchers require `bucket` to be an absolute filesystem path prefix
- Local filesystem fetchers require `key` to be a relative path suffix (not absolute)
- Local filesystem fetchers must reject keys that escape the bucket prefix after path resolution (for example `../` traversal)
- File fetchers support partial success: valid entries return staged objects while invalid entries return per-entry dead letters
- Legacy inventory sources are out-of-scope for this contract shift; only active orchestrated sources must satisfy the new identity contract
- `ExtractionResult` is the single carrier of identity and unstructured payload between transform and load
- Logs/artifacts/manifests emit identity as explicit `bucket`, `key`, `version_id` fields
- `facility` and `hash` are derived once during transform/extraction and carried to sinks; sinks must not re-derive
- `UnstructuredMetadata.metadata` is a JSON string in the shared contract; sinks persist it without re-derivation
- Unstructured writes use latest-write-wins semantics on hash; duplicate hashes in a write batch keep the last row
- Version-pinned fetch target rendering is centralized in a shared helper used by all fetchers
- The `StructuredSink` enforces the Structured Metadata schema on write
- `StructuredMetadata` dataclass is schema source-of-truth for Polars, PyArrow, and Iceberg representations
- `UnstructuredMetadata` dataclass is schema source-of-truth for Polars, PyArrow, and Iceberg representations, including `schema_version`
- Structured and unstructured S3 tables publish schema-version table properties; breaking schema changes must bump version
- Unstructured schema transitions that introduce new identity/join fields use reset-and-reload, not in-place backfill
- **File Format** is determined from magic bytes (first 8 bytes of the file), not from xarray internals — avoids coupling to private xarray/netCDF4 attributes
- NetCDF4/HDF5 files require multiple block fetches to traverse the HDF5 btree even for metadata-only reads; this makes `S3XarrayHandle` slower for large NetCDF4 files than for NetCDF3
- **Three-layer concurrency model**: peak resource usage is `batch_workers × transform_threads × s5cmd_workers`. On Fargate each worker runs in an isolated container so all three can be maximised independently. For local runs all three layers share the same machine and must be capped together to avoid memory exhaustion and CPU saturation.

## Releases

**Release**:
A versioned snapshot of the package published as a GitHub Release. Triggered by pushing a Git tag matching `vMAJOR.MINOR.PATCH` (e.g. `v1.2.3`). The release workflow: runs lint + tests, builds a wheel and sdist using `hatch-vcs` (version derived from the tag), attaches them to the GitHub Release with an auto-generated changelog. Installable via a direct URL from the release page.
_Avoid_: deploy, publish, ship (use "release" consistently)

**CI Workflow**:
A GitHub Actions workflow that runs on every pull request to `main`: lint (`ruff check` + `ruff format --check`) and the full test suite (`pytest`). Passing CI is a required status check — PRs cannot be merged until CI passes.
_Avoid_: build pipeline, validation pipeline

## Example dialogue

> **Dev:** "Should we re-index a file if we've already seen it?"
> **Domain expert:** "Yes. Re-indexing the same object version is safe — the S3-table sinks upsert by **Object Reference Hash** (derived from **Object Version Identity**), so the latest write replaces earlier values for that version."

> **Dev:** "Why do we keep **Unstructured Metadata** in memory through `transform` instead of persisting intermediate handles?"
> **Domain expert:** "In our current batch sizing, in-memory payloads are acceptable and the simpler contract parity with **Structured Metadata** is more valuable than an extra persistence abstraction."

> **Dev:** "Why doesn't the extractor use `geospatial_lat_min` from global attributes instead of computing min/max from the coordinate arrays?"
> **Domain expert:** "We don't know yet which files carry those ACDD attributes reliably. The first pipeline run will collect **Unstructured Metadata** for every file — we'll analyse the attribute distribution and build a v2 extractor once we know what's actually there."

> **Dev:** "Why does `ThresholdFileFetcher` default unknown file sizes to the cloud path?"
> **Domain expert:** "If we don't know the size, downloading could be arbitrarily expensive. Cloud byte-range reads are safe for any size — they just cost more per-request for very small files."

## Flagged ambiguities

- "metadata" (unqualified) was used to mean both **Structured Metadata** and **Unstructured Metadata** — resolved: always qualify the term.
- "collection" and "facility" were both used for the second S3 path segment — resolved: canonical domain term is **Facility**; remove `collection` from pipeline schemas.
- "file format" was initially derived from xarray private internals (`_file_obj._ds.file_format`) — resolved: always read from magic bytes via `XarrayHandle.file_format`.
- "`s3_uri` was used as a complete identity" — resolved: canonical identity is **Object Version Identity** (`bucket`, `key`, `version_id`).
- "`version_id` could be null" — resolved: identity fields are required and non-null; data/contracts must enforce this.
- "Could local files omit `version_id`?" — resolved: local filesystem inputs use `version_id = __LOCAL__` (non-null sentinel).
- "Should local filesystem identity be (`logical bucket`, absolute `key`) or (absolute `bucket` prefix, relative `key` suffix)?" — resolved: local identity uses absolute `bucket` prefix up to anchor + relative `key` suffix after anchor.
- "Should local fetchers accept arbitrary `version_id` values?" — resolved: no; local fetchers require `version_id = __LOCAL__`.
- "Should local fetchers ignore `bucket`?" — resolved: yes for validation; local fetchers do not compare to configured bucket values and only resolve local file paths from `bucket + key`.
- "`s3_uri` was treated as a required pipeline identifier" — resolved: remove `s3_uri` from pipeline contracts and use **Object Version Identity** only.
- "S3 identity field names were ambiguous (`s3_*` vs unprefixed)" — resolved: use `bucket`, `key`, `version_id` consistently across pipeline contracts.
- "Should local inventory filtering use grep or glob?" — resolved: use glob-based path discovery (`root_path` + pattern), not grep.
- "Should local inventory subset always include all matches?" — resolved: default is all matches; optional deterministic `max_files` cap is allowed.
- "Do we need a separate local sink class?" — resolved: no; use `IcebergTableSink` with `SqliteCatalogConfig`.
- "How to hand off identity between stages (composite fields vs value object)" — resolved: use a lightweight named-tuple **Object Reference**.
- "`UnstructuredMetadata` referred to both row contract and persisted handle type" — resolved: keep `UnstructuredMetadata` as the row contract and remove the separate handle abstraction.
- "`UnstructuredMetadata.metadata` payload type (dict vs JSON string) was ambiguous" — resolved: canonical shared contract is JSON string payload.
- "Should unstructured payloads use intermediate diskcache handles or stay in-memory between transform and load?" — resolved: keep `UnstructuredMetadata` rows in memory and remove handle/cache abstractions.
- "Should sink joins use composite identity or hash?" — resolved: use **Object Reference Hash** for joins/upserts, while keeping explicit **Object Version Identity** fields required everywhere.
- "Should DynamoDB use `facility` or `hash` as the base partition key?" — resolved: base key is `hash`; `facility` is an index access path.
- "Should DynamoDB facility replay scan full partitions or drain incrementally?" — resolved: incremental draining with `indexed_at_ms` ordering.
- "Missing facility handling (fail vs nullable vs sentinel)" — resolved: use sentinel `UNKNOWN`.
- "Batch partitioner naming used `collection`" — resolved: rename to `FacilityGroupedBatchPartitioner`.
- "Should transform/load exchange Arrow tables instead of domain rows?" — deferred: keep typed domain-row contracts for now; revisit later.
- "Should local Parquet sinks match hash-upsert parity with S3 sinks?" — deferred: S3-path parity now; local Parquet parity later.
