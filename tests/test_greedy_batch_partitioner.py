import dataclasses

import polars

from data_index.batch_partitioner.greedy import GreedyBatchPartitioner


def _inventory(*sizes: int) -> polars.DataFrame:
    return polars.DataFrame(
        {
            "bucket": ["bucket" for _ in range(len(sizes))],
            "key": [f"file{i}.nc" for i in range(len(sizes))],
            "version_id": [f"v{i}" for i in range(len(sizes))],
            "size": list(sizes),
        }
    )


def test_all_files_fit_in_one_batch():
    partitioner = GreedyBatchPartitioner(max_files=10, max_bytes=100)
    batches = list(partitioner.partition(_inventory(10, 20, 30)))

    assert len(batches) == 1
    assert len(batches[0]) == 3


def test_size_limit_splits_into_multiple_batches():
    partitioner = GreedyBatchPartitioner(max_files=100, max_bytes=50)
    # 30+30 > 50, so each file gets its own batch
    batches = list(partitioner.partition(_inventory(30, 30, 30)))

    assert len(batches) == 3
    assert all(len(b) == 1 for b in batches)


def test_file_count_limit_splits_into_multiple_batches():
    partitioner = GreedyBatchPartitioner(max_files=2, max_bytes=1000)
    batches = list(partitioner.partition(_inventory(10, 10, 10, 10, 10)))

    assert len(batches) == 3
    assert len(batches[0]) == 2
    assert len(batches[1]) == 2
    assert len(batches[2]) == 1


def test_empty_inventory_yields_no_batches():
    partitioner = GreedyBatchPartitioner(max_files=10, max_bytes=100)
    batches = list(partitioner.partition(_inventory()))

    assert batches == []


def test_batches_contain_all_original_files():
    partitioner = GreedyBatchPartitioner(max_files=2, max_bytes=1000)
    inventory = _inventory(10, 20, 30, 40)
    batches = list(partitioner.partition(inventory=inventory))

    object_references = [
        {
            k: v
            for k, v in dataclasses.asdict(object_reference).items()
            if k not in {"xarray_handle", "extraction_result"}
        }
        for batch in batches
        for object_reference in batch
    ]
    assert inventory.equals(
        other=polars.DataFrame(data=object_references),
    )
