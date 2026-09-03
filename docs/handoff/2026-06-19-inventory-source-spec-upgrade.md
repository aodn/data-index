# Handoff: inventory-source-spec-upgrade

## Scope for next session
Implement the InventorySource spec upgrade so inventory rows include `last_modified`, `size`, and `etag`, and keep the pipeline stable.

## Current state
- Structured metadata/schema version wiring has already been implemented and validated in this session.
- Repository currently has uncommitted changes in:
  - `src/data_index/defaults/local.py`
  - `src/data_index/structured_metadata.py`
  - `src/data_index/structured_sink/s3_table_sink.py`
  - `tests/test_structured_metadata.py`
  - `tests/test_structured_parquet_sink.py`
  - `tests/test_structured_s3_table_sink.py`
- Last test run result in this session: full suite passed (`113 passed`).

## Findings from investigation (inventory spec upgrade)
The new inventory fields require coordinated updates across protocol, sources, extract validation, and tests.

### Required code touchpoints
1. Protocol + docs
   - `src/data_index/protocols.py` (`InventorySource.inventory` docstring/contract)
   - `CONTEXT.md` references to inventory columns
   - Inventory source docstrings currently mention only `s3_uri` + `size`

2. Inventory source implementations
   - `src/data_index/inventory_source/live_s3.py`
   - `src/data_index/inventory_source/live_s3_facility_subset.py`
   - `src/data_index/inventory_source/s3_table.py`
   - `src/data_index/inventory_source/parquet.py`
   - Add/select/output `last_modified` + `etag` with consistent names
   - Map upstream names:
     - `last_modified_date` -> `last_modified`
     - `e_tag` -> `etag`

3. Scan/projection config
   - `src/data_index/iceberg_config/table_scan_config.py`
   - Ensure `selected_fields` includes upstream columns needed to derive `last_modified` and `etag`

4. Extract boundary / batch schema checks
   - `src/data_index/extract.py`
   - Current schema validation is strict and only accepts `s3_uri` + `size`
   - Update to accept upgraded inventory schema (or superset validation while still requiring required fields)

5. Tests needing updates
   - `tests/test_live_s3_inventory_source.py`
   - `tests/test_live_s3_facility_subset_inventory_source.py`
   - `tests/test_s3_table_inventory_source.py`
   - `tests/test_parquet_inventory_source.py`
   - Any tests asserting exact inventory column set/order and/or strict extract schema assumptions

## Compatibility decision to make early
Decide whether old cached inventory parquet files (missing new columns) should:
- fail fast, or
- be accepted with null defaults for `last_modified`/`etag` during transition.

Recommended in dev: support null defaults briefly to avoid friction while sources/tests are updated.

## Related artifacts
- Session plan/progress notes: `/Users/thommodin/.copilot/session-state/4241b316-e24b-4e85-83a3-1e78754b0303/plan.md`

## Suggested skills for next session
- `tdd` (recommended): safest way to roll out the cross-cutting schema/contract change.
- `to-issues` (optional): if you want this broken into separate implementation tickets.
