from __future__ import annotations

import typing

import polars
import pydantic


class GreedyBatchPartitioner(pydantic.BaseModel):
    """Splits an inventory DataFrame into Batches using greedy bin-packing.

    Each Batch satisfies both max_files (file count) and max_bytes (total size)
    limits. Files are packed into the current batch until either limit is hit,
    then a new batch is started.
    """

    max_files: int
    max_bytes: int

    def partition(
        self, inventory: polars.DataFrame
    ) -> typing.Iterator[polars.DataFrame]:
        current_rows: list[dict] = []
        current_bytes = 0

        for row in inventory.iter_rows(named=True):
            size = row["size"] or 0
            if current_rows and (
                len(current_rows) >= self.max_files
                or current_bytes + size > self.max_bytes
            ):
                yield polars.DataFrame(current_rows, schema=inventory.schema)
                current_rows = []
                current_bytes = 0
            current_rows.append(row)
            current_bytes += size

        if current_rows:
            yield polars.DataFrame(current_rows, schema=inventory.schema)
