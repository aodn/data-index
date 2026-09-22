# Handoff: dynamodb-write-path-optimization

## Scope for next session
Implement the DynamoDB write-path and replay-index design that was finalized in this session (key strategy, replay access pattern, and operational guards), then validate with focused tests.

## Source-of-truth artifacts (already captured)
- `docs/adr/0011-dynamodb-hash-primary-key-with-facility-replay-index.md`
- `CONTEXT.md` (DynamoDB sink constraints/ambiguities and replay decisions)

Use those files for rationale and final decisions; avoid re-deriving design choices.

## What was implemented in code this session
- Added `src/data_index/sink/dynamodb_sink.py`
  - Supports structured + unstructured metadata rows
  - Base table key is `hash`
  - Provisioning, batched writes, retries, and typed item serialization
  - Added `row_kind` attribute for operational visibility
- Wired sink for runtime typing/export:
  - `src/data_index/sink/__init__.py`
  - `src/data_index/runners/types.py`
- Added optional defaults (opt-in, not switched as runtime default):
  - `src/data_index/runners/defaults.py`
- Added/updated documentation:
  - `README.md`
  - `CONTEXT.md`
- Added tests:
  - `tests/test_dynamodb_sink.py`

## Validation run
- `uv run ruff check src/data_index/sink/dynamodb_sink.py src/data_index/runners/defaults.py tests/test_dynamodb_sink.py`
- `uv run pytest tests/test_dynamodb_sink.py -q`

## Priority implementation gaps (next session)
1. **Replay index provisioning**
   - Extend `DynamoDBSink.provision()` to create facility replay GSI(s) per ADR/CONTEXT decisions.
2. **Replay ordering attribute**
   - Add `indexed_at_ms` on write path and ensure it is present/typed consistently for both structured and unstructured rows.
3. **Row-level oversize protection**
   - Primary guard in transform path for unstructured payload size (dead-letter only offending rows).
   - Sink-level backstop guard before `batch_write_item`.
4. **Retry behavior**
   - Switch write retry strategy to exponential **full jitter**.
5. **Replay consumer contract**
   - Implement/query pattern that drains by facility incrementally (checkpointed, at-least-once) and batch-gets full rows.
6. **Tests**
   - Add coverage for GSI config/provision behavior, `indexed_at_ms` population, oversize dead-letter routing, and jitter/backoff semantics.

## Suggested skills for next session
- `tdd` (recommended): drive the remaining write-path changes and replay-index behavior safely.
- `diagnose`: if replay/drain throughput or throttling behavior is inconsistent under concurrency.
