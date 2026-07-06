# DuckDB as an opt-in write engine for S3 Tables sinks

We are keeping `pyiceberg` as the default sink write engine and adding `duckdb` as an explicit per-sink opt-in for Iceberg upserts via attached S3 Tables catalogs. This reverses ADR-0007's earlier DuckDB rejection because the current target runtime now supports the required `MERGE`-based upsert workflow, while preserving existing behavior by default. DuckDB write mode is constrained to S3 Tables catalogs in this iteration; non-S3 catalog DuckDB wiring remains out of scope.
