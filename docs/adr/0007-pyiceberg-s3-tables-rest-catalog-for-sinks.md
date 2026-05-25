# PyIceberg with S3 Tables REST catalog for sink implementations

The production `StructuredSink` and `UnstructuredSink` write to AWS S3 Tables (Apache Iceberg) using PyIceberg as the client. We chose the native S3 Tables REST catalog endpoint over AWS Glue Data Catalog to avoid a Glue dependency — S3 Tables exposes a first-class Iceberg REST catalog that PyIceberg supports directly. DuckDB's Iceberg write support was considered but its S3 Tables catalog integration is immature. PySpark was rejected as too heavyweight for a Fargate-based pipeline.

## Considered options

- **PyIceberg + S3 Tables REST catalog** — chosen: native interface, no extra AWS services, clean PyIceberg support.
- **PyIceberg + AWS Glue catalog** — rejected: adds a Glue dependency with no benefit for this use case.
- **DuckDB** — rejected: S3 Tables catalog integration is not production-ready.
- **PySpark** — rejected: requires a Spark cluster; incompatible with the Fargate single-container model.
