# Handoff — data-index cluster orchestration design

## Repo / context

- Repo: `aodn/data-index`
- Branch: (current working branch)
- Working directory: `/Users/thommodin/dev/data-index`
- Domain doc: `CONTEXT.md` — **updated this session** (see below)
- ADRs: `docs/adr/` — **ADR-0006 added this session**

## What was done this session

### Design session — Orchestrator run environment

Conducted a `grill-with-docs` session to design the cluster orchestration system using the `src/data_index/cluster/` tooling. All decisions are captured in:

- **`CONTEXT.md`** — four new terms added under a new `## Cluster Run` section:
  - `Orchestrator`, `InventorySource`, `BatchPartitioner`, `Run State File`
- **`docs/adr/0006-s3-state-file-for-run-resumption.md`** — records why we use an S3 JSON state file instead of Prefect task caching for run resumption (payload size cost)

### Decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Orchestrator runs locally (for now) | Local, ECS/EC2 long-term | Defer infra setup |
| Inventory source | Injectable `InventorySource` protocol | `ParquetInventorySource` for tests, `LiveS3InventorySource` for production |
| Chunking | Injectable `BatchPartitioner` protocol | `GreedyBatchPartitioner` first; `CollectionGroupedBatchPartitioner` later for optimisation |
| Worker unit | One `@prefect.task` per Batch, full ETL inside | Avoids coordination complexity |
| Sinks | Injectable (existing `StructuredSink`/`UnstructuredSink`) | Parquet-to-S3 for now, S3 Tables long-term |
| Batch dispatch | Serialize DataFrame directly via Dask | ~20MB per batch is acceptable; avoids S3 manifest indirection |
| Worker sizing | Runtime parameters | Configurable `n_workers`, `worker_cpu`, `worker_mem` |
| Run resumption | S3 JSON state file | See ADR-0006 |
| Error handling | Prefect retries on batch task | ETL is idempotent per file |
| Batch size | 10k–100k files, ≤50GB | Already enforced in `extract.py` |

### Corpus scale

~20M NetCDF files → ~200 Batches at 100k files/batch.

## Agreed module layout (not yet implemented)

```
src/data_index/protocols.py            ← add InventorySource, BatchPartitioner protocols
src/data_index/inventory_source/       ← ParquetInventorySource, LiveS3InventorySource
src/data_index/batch_partitioner/      ← GreedyBatchPartitioner, (later) CollectionGroupedBatchPartitioner
src/data_index/cluster/orchestrate.py  ← the Orchestrator Prefect flow
```

## Current state

- `src/data_index/cluster/` has: `main.py` (prototype), `fargate_cluster_config.py`, `docker_image.py`, `Dockerfile`
- None of the new modules above have been created yet
- All existing tests still pass (`uv run pytest tests/ -v`)

## Suggested next steps

1. **Add `InventorySource` and `BatchPartitioner` to `src/data_index/protocols.py`**
2. **Implement `ParquetInventorySource`** and **`GreedyBatchPartitioner`** (the first concrete implementations)
3. **Implement `src/data_index/cluster/orchestrate.py`** — the Orchestrator Prefect flow that:
   - Reads inventory via `InventorySource`
   - Partitions into Batches via `BatchPartitioner`
   - Reads/writes Run State File from S3 to skip completed Batch IDs
   - Dispatches remaining Batches via `DaskTaskRunner` to Fargate workers
   - Each worker task calls `extract → transform → load` with injected sinks
4. **Tests** for `GreedyBatchPartitioner` (size and count limits) and `ParquetInventorySource`
5. **Defer:** `LiveS3InventorySource`, `CollectionGroupedBatchPartitioner`, ECS/EC2 orchestrator hosting

## Suggested skills for next session

- `tdd` — to implement and test `InventorySource`, `BatchPartitioner`, and the Orchestrator flow red-green-refactor style
