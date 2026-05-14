# Handoff — data-index DI refactor

## Repo / context

- Repo: `aodn/python-spike-testing`
- Working directory: `/Users/thommodin/dev/python-spike-testing/data-index`
- Domain doc: `data-index/CONTEXT.md`
- ADRs: `data-index/docs/adr/`

### Key decisions (see ADR `0002`)

- `data-index/docs/adr/0002-protocol-based-di-for-pipeline-components.md`

### Structural changes

**New file:** `src/data_index/protocols.py`
- `ManifestEntry` Pydantic model (`s3_uri`, `absolute_path`)
- `StructuredMetadata` dataclass with `polars_schema: typing.ClassVar[polars.Schema]`
- `ExtractionResult` dataclass (`structured_metadata: StructuredMetadata | None`)
- `FileFetcher` protocol → `fetch(uris, extract_path) -> list[ManifestEntry]`
- `MetadataExtractor` protocol → `extract(ds, s3_uri) -> ExtractionResult`
- `StructuredSink` protocol → `write(data: list[StructuredMetadata]) -> None`
- `UnstructuredSink` protocol → `write(data: dict[str, dict]) -> None`

**Implementation directories** (per-protocol subdirectories):
- `src/data_index/file_fetcher/`
  - `s3_fetcher.py` — `S3Fetcher` (boto3, lazy client in `fetch()`)
  - `s5cmd_fetcher.py` — `S5CMDFetcher` (s5cmd via `sh`, default impl)
- `src/data_index/metadata_extractor/`
  - `netcdf_extractor.py` — `NetCDFExtractor` (xarray)
- `src/data_index/structured_sink/`
  - `parquet_sink.py` — `ParquetSink` (uses `StructuredMetadata.polars_schema`)
- `src/data_index/unstructured_sink/`
  - `diskcache_sink.py` — `DiskCacheSink`

**Refactored pipeline:**
- `extract.py` — accepts `FileFetcher`, returns `list[ManifestEntry]`
- `transform.py` — accepts `MetadataExtractor` + `UnstructuredSink`, returns `list[StructuredMetadata]`
- `load.py` — accepts `StructuredSink`, calls `sink.write(structured_metadata)`
- `main.py` — wires all four concrete impls (`S5CMDFetcher` as default fetcher, `NetCDFExtractor`, `ParquetSink`, `DiskCacheSink`)

### Key architectural notes

- `UnstructuredSink` is injected into `transform` (not `load`) — avoids serialising large unstructured dicts as Prefect task results and preserves incremental write durability
- `S5CMDFetcher` is the default fetcher (uses `s5cmd` via `sh`); `S3Fetcher` (boto3) is an available alternative with lazy client creation in `fetch()` — safe for Prefect cloudpickle
- `STRUCTURED_METADATA_SCHEMA` removed from protocols; now lives as `StructuredMetadata.polars_schema` class var
- No LazyFrames cross task boundaries (they don't avoid Prefect serialisation for in-memory data)

## State

All imports verified clean:
```
uv run python -c "from data_index.main import pipeline; print('OK')"
```

No tests written yet for the new protocol/impl structure.

## Suggested next steps

1. Write unit tests for each impl class (`S3Fetcher`, `S5CMDFetcher`, `NetCDFExtractor`, `ParquetSink`, `DiskCacheSink`) using test doubles that satisfy the protocols
2. Write integration test for `pipeline()` with mock impls
3. Implement `DynamoDBSink` as a second `UnstructuredSink` (currently only `DiskCacheSink` exists)
4. Implement a `S3TableSink` as a second `StructuredSink` (currently only `ParquetSink` exists)

## Suggested skills for next session

- `tdd` — for writing tests against the new protocol boundaries
- `caveman` — active this session, use if preferred
