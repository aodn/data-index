# Local subset metadata analysis handoff

## Scope for next session

Design + implement analysis for outputs of:

- `src/data_index/defaults/local_subset.py`
- `src/data_index/metadata_extractor/attribute_netcdf_extractor.py`

User asked for handoff focused on continuing this analysis work.

## What is already resolved

1. Canonical term is **Facility** (not collection) for second S3 path segment (`IMOS/<facility>/...`).
2. `collection` remains legacy field/schema name where already present in code/storage.
3. Analysis metrics to deliver:
   - Facility extraction success %
   - Structured per-field success %
   - Unstructured mapping correctness diagnostics
4. Because current outputs only persist succeeded rows, extraction success % must be **estimated** (no pipeline code change approved).

## Decisions captured in repository docs

- Updated glossary + ambiguity resolution in:
  - `CONTEXT.md`

Key glossary decision: **Facility** is canonical domain term; `collection` is legacy alias.

## Agreed analysis design

1. **Facility extraction success % (estimator)**
   - Denominator per facility: `min(subset_per_facility, inventory_count_facility)` where `subset_per_facility = 10_000`, derived from inventory parquet under `.extract/s3_metadata` (same sampling basis as `LiveS3InventorySourceFacilitySubset`).
   - Numerator per facility: distinct `s3_uri` count in successful outputs.
   - Success % = numerator / denominator.

2. **Structured per-field success**
   - From `structured_metadata.parquet`.
   - Produce global + per-facility `% non-null` for each structured field.

3. **Unstructured mapping correctness**
   - Compare `attributes_map` in `AttributeNetCDFExtractor` against keys in unstructured `global_attrs`.
   - Produce global + per-facility metrics per mapped field:
     - exact key-presence %
     - populated-structured %
     - cast-fail proxy % for numeric fields (`lat/lon`): source key present but structured value null
     - case-mismatch signals (`lower(key)` matches only)
     - fuzzy key suggestions (normalized + score >= 90 using `thefuzz`/rapidfuzz)

4. **Flagging/reporting**
   - Include raw metrics + threshold-based flags + fuzzy suggestions together.
   - Threshold rule approved: flag where `key_presence >= 20%` and `populated_structured <= 5%`.

## Constraints and user choices

- User explicitly rejected pipeline code change to persist status ledger.
- User explicitly wants both threshold flags and raw outputs.
- User approved fuzzy matching suggestions for possible key mapping fixes.

## Suggested next concrete steps

1. Create analysis artifact (single script or notebook) that reads:
   - `.extract/s3_metadata/**/*.parquet` (inventory basis)
   - local subset outputs (`structured_metadata.parquet`, `unstructured_metadata.parquet`)
2. Implement three output tables/reports:
   - facility success estimator table
   - structured per-field coverage (global + facility)
   - mapping mismatch report (exact/case/fuzzy + cast-fail proxy)
3. Validate with small sample first, then full local subset output.

## Suggested skills for next session

- `prototype` (recommended): quickly build runnable analysis script/notebook variations and compare output shape.
- `diagnose`: if mismatches or metric anomalies appear during implementation.
- `grill-with-docs`: if terminology or glossary decisions need further tightening while coding.
