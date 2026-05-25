# S3 state file for Orchestrator run resumption

The Orchestrator runs against ~20M files split into ~200 Batches. A full run takes many hours, so resumption after failure is essential. We track completed Batch IDs in a JSON file written to S3 rather than using Prefect's built-in task caching (`persist_result=True`).

Prefect result persistence stores the task's return value — for a Batch task that processes 100k files, the return value (a list of `ExtractionResult` objects) would be tens of megabytes per task, and Prefect charges per stored result. An S3 JSON file containing a set of completed Batch IDs is effectively free and has no per-entry cost. The Batch ID is a deterministic hash of the sorted `s3_uri` list, so the state file survives across re-deployments.

**Considered alternative:** Prefect task caching keyed by Batch hash. Rejected because the result payload per Batch is large and Prefect state storage costs scale with payload size, not number of tasks.
