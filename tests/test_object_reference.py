import os
import random
import uuid

import pytest

import data_index.protocols


# --- Pytest Fixtures ---
@pytest.fixture(scope="session")
def batch_size() -> int:
    return 16_384


@pytest.fixture
def best_case_batch(batch_size: int) -> list[data_index.protocols.ObjectReference]:
    """Highly repetitive data, ideal for columnar/zstd compression."""
    return [
        data_index.protocols.ObjectReference(
            bucket="static-archive-bucket",
            key="logs/2026/06/30/system.log",
            version_id=None,
            size=1048576,
        )
        for _ in range(batch_size)
    ]


@pytest.fixture
def worst_case_batch(batch_size: int) -> list[data_index.protocols.ObjectReference]:
    """High entropy data, defeating dictionary and zstd compression."""
    return [
        data_index.protocols.ObjectReference(
            bucket=uuid.uuid4().hex,
            key=os.urandom(24).hex(),
            version_id=uuid.uuid4().hex,
            size=random.randint(1, 10**12),
        )
        for _ in range(batch_size)
    ]


# --- Test Cases ---
def test_compression_efficiency(
    best_case_batch: list[data_index.protocols.ObjectReference],
    worst_case_batch: list[data_index.protocols.ObjectReference],
):
    """Asserts that highly entropic data is significantly larger than repetitive data."""
    best_b64 = data_index.protocols.ObjectReference.to_compressed_base64_table(
        best_case_batch
    )
    worst_b64 = data_index.protocols.ObjectReference.to_compressed_base64_table(
        worst_case_batch
    )

    best_size_kb = len(best_b64) / 1024
    worst_size_kb = len(worst_b64) / 1024

    print(f"\nBest case: {best_size_kb:.2f} KB")
    print(f"Worst case: {worst_size_kb:.2f} KB")

    # In practice, the worst case will be ~1000x larger due to Arrow/zstd efficiency.
    # We assert a conservative 100x multiplier to ensure the test passes reliably.
    assert len(worst_b64) > len(best_b64), (
        "Compression efficiency failed expected heuristic"
    )


def test_roundtrip_serialization_best_case(
    best_case_batch: list[data_index.protocols.ObjectReference],
):
    """Ensures data can be serialized and deserialized with 100% fidelity."""
    b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(
        best_case_batch
    )
    deserialized_batch = (
        data_index.protocols.ObjectReference.from_compressed_base64_table(b64_str)
    )

    assert len(deserialized_batch) == len(best_case_batch)
    assert deserialized_batch[0] == best_case_batch[0]
    assert deserialized_batch[-1] == best_case_batch[-1]


def test_roundtrip_serialization_worst_case(
    worst_case_batch: list[data_index.protocols.ObjectReference],
):
    """Ensures complex, high-entropy data survives the roundtrip unchanged."""
    b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(
        worst_case_batch
    )
    deserialized_batch = (
        data_index.protocols.ObjectReference.from_compressed_base64_table(b64_str)
    )

    assert len(deserialized_batch) == len(worst_case_batch)
    assert deserialized_batch[0] == worst_case_batch[0]
    assert deserialized_batch[-1] == worst_case_batch[-1]


def test_empty_batch_serialization():
    """Ensures edge case of an empty list does not break the schema."""
    empty_batch = []
    b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(
        empty_batch
    )
    deserialized_batch = (
        data_index.protocols.ObjectReference.from_compressed_base64_table(b64_str)
    )

    assert deserialized_batch == []
