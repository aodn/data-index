# Pipeline

```
Batch (S3 URIs + sizes)
  |
  v
extract()  <--  FileFetcher
  |
  v
Manifest (s3_uri + local path)
  |
  v
transform()  <--  MetadataExtractor
  |               UnstructuredMetadata (diskcache write)
  v
ExtractionResult (structured row + unstructured ref + status)  [x N files]
  |
  v
load()
  ├─ StructuredSink    -->  Parquet / S3 Table (not implemented)
  └─ UnstructuredSink  -->  Parquet / DynamoDB (Not Implemented) / S3 Table (not implemented)
```

# Running

## Run the uv server
```bash
uv run prefect server start
```

## Run the pipeline
```bash
uv run data-index
```