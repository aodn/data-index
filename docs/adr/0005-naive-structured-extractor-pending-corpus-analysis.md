# Naive structured extractor pending corpus analysis

The IMOS corpus is a mixture of NetCDF3 and NetCDF4/HDF5 files with unknown ACDD attribute coverage. A production `MetadataExtractor` would read spatial and temporal extent from CF-ACDD global attributes (`geospatial_lat_min`, `time_coverage_start`, etc.) at zero I/O cost, rather than computing min/max from coordinate arrays. However, we do not yet know what fraction of files carry these attributes reliably.

The current `NetCDFExtractor` is intentionally naive: it derives extent by computing lazy reductions over coordinate arrays (`.min()/.max()` — not `.values.min()`, which materialises the full array). The unstructured metadata collected in the first pipeline run will contain all global attributes for every file. Analysis of that corpus will reveal the actual ACDD attribute distribution and inform a v2 structured extractor that can use attributes where available and fall back to coordinate reduction only where necessary.

## Considered options

- **ACDD-first with fallback now** — rejected: we don't know the distribution, so we can't write a reliable fallback strategy without guessing.
- **Always use coordinate reduction** — current approach; consistent across all file formats and conventions, just slower on the cloud path for large coordinate arrays.
