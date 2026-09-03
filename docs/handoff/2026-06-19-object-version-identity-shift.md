# Handoff: object-version-identity-shift

## Scope for next session
Implement the protocol and schema migration from `s3_uri` identity to canonical object-version identity (`bucket`, `key`, `version_id`) across active orchestrated paths.

## Artifacts updated in this session
- `CONTEXT.md` (language, relationships, and constraints were updated inline during grilling)
- `docs/adr/0009-object-version-identity-for-upserts.md`

For the full rationale/decision record, use those two files as source of truth rather than this handoff.

## Key outcomes to carry forward
- Design grilling is complete and decisions are locked for this migration.
- The repo now has a new ADR for this change (`0009`).
- This session did **not** implement code changes beyond documentation/ADR updates.

## Implementation touchpoints (code)
1. `src/data_index/protocols.py`
   - Introduce `ObjectReference` as `typing.NamedTuple`
   - Replace `s3_uri`-based identity in `BatchEntry`, `RawExtractionResult`, `ExtractionResult`, `XarrayHandle`, and sink contracts
2. `src/data_index/extract.py`
   - Validate required columns: `bucket`, `key`, `version_id`, `size` (allow extras)
   - Enforce duplicate checks on full triple identity
3. `src/data_index/transform.py`, `src/data_index/load.py`, `src/data_index/orchestrate.py`
   - Carry `object_ref` end-to-end
   - Keep identity on `ExtractionResult`; `UnstructuredMetadata` remains payload-only
4. `src/data_index/file_fetcher/{threshold_fetcher.py,s3_fetcher.py,s5cmd_fetcher.py}`
   - Use shared helper for version-pinned fetch target rendering
   - Ensure fetches read exact `version_id`
5. `src/data_index/xarray_handle/{s3_xarray_handle.py,disk_xarray_handle.py}`
   - Replace `s3_uri` identity surface with `object_ref`
6. `src/data_index/unstructured_metadata/disk_cache_unstructured_metadata.py`
   - Cache key becomes tuple `(bucket, key, version_id)`
7. `src/data_index/structured_metadata.py`
   - Replace `s3_uri` with required non-null `bucket`, `key`, `version_id`
   - Replace `collection` with `facility`
   - Bump schema version
8. `src/data_index/metadata_extractor/{netcdf_extractor.py,attribute_netcdf_extractor.py}` and `src/data_index/_collection.py`
   - Derive `facility` once from `key`, coercing missing/invalid to `UNKNOWN`
   - Populate identity/facility in extracted rows
9. `src/data_index/structured_sink/s3_table_sink.py` and `src/data_index/unstructured_sink/s3_table_sink.py`
   - Upsert join columns: `bucket`, `key`, `version_id`
   - Partitioning: structured by `facility` + year; unstructured by `facility`
   - Add explicit opt-in reset mode in `provision()`
   - Ensure schema-version table properties for both tables
10. Inventory sources (active only): `src/data_index/inventory_source/{live_s3.py,live_s3_facility_subset.py,s3_table.py,parquet.py}` and scan config `src/data_index/iceberg_config/table_scan_config.py`
   - Emit/pass required identity columns with non-null contract
   - Keep legacy sources out of scope
   - Preserve injectability for inventory behavior (latest-only vs all versions)
11. Batch/run state
   - Update any batch-ID hashing logic to use sorted (`bucket`, `key`, `version_id`) identities
12. Naming alignment
   - Rename `CollectionGroupedBatchPartitioner` to `FacilityGroupedBatchPartitioner` (optionally keep alias if needed)

## Tests expected to change
- `tests/test_live_s3_inventory_source.py`
- `tests/test_live_s3_facility_subset_inventory_source.py`
- `tests/test_s3_table_inventory_source.py`
- `tests/test_parquet_inventory_source.py`
- `tests/test_structured_metadata.py`
- `tests/test_structured_s3_table_sink.py`
- `tests/test_unstructured_s3_table_sink.py`
- `tests/test_threshold_fetcher.py`
- `tests/test_s3_fetcher.py`
- `tests/test_s5cmd_fetcher.py`
- `tests/test_disk_cached_unstructured_metadata.py`
- `tests/test_netcdf_extractor.py`
- Any tests asserting `s3_uri` identity or `collection` fields

## Suggested skills for next session
- `tdd` (recommended): safest way to drive this broad contract migration.
- `diagnose`: useful if version-pinned fetch behavior differs across S3 backends.
