from __future__ import annotations

import typing

import polars
import pydantic

from data_index.protocols import ObjectReference


class GreedyBatchPartitioner(pydantic.BaseModel):
    """Splits an inventory DataFrame into Batches using greedy bin-packing.

    Each Batch satisfies both max_files (file count) and max_bytes (total size)
    limits. Files are packed into the current batch until either limit is hit,
    then a new batch is started.
    """

    type: typing.Literal["greedy_batch_partitioner"] = pydantic.Field(
        default="greedy_batch_partitioner"
    )

    max_files: int
    max_bytes: int

    def partition(
        self, inventory: polars.DataFrame
    ) -> typing.Iterator[list[ObjectReference]]:
        if inventory.is_empty():
            return

        # This creates a cheap, lazy-like view without copying the underlying dataframe memory.
        inventory = inventory.select(
            [
                "bucket",
                "key",
                "version_id",
                polars.col("size").fill_null(0),
            ]
        )

        # Construct the ObjectReference generator
        object_reference_generator = (
            ObjectReference(
                bucket=bucket,
                key=key,
                version_id=version_id,
                size=size,
            )
            for bucket, key, version_id, size in inventory.select(
                polars.col("bucket"),
                polars.col("key"),
                polars.col("version_id"),
                polars.col("size").fill_null(value=0),
            ).iter_rows(named=False)
        )

        # Stream the objects one-by-one into batches
        current_batch: list[ObjectReference] = []
        current_bytes = 0

        for obj in object_reference_generator:
            size = obj.size or 0

            if current_batch and (
                len(current_batch) >= self.max_files
                or current_bytes + size > self.max_bytes
            ):
                yield current_batch
                current_batch = []
                current_bytes = 0

            current_batch.append(obj)
            current_bytes += size

        if current_batch:
            yield current_batch
