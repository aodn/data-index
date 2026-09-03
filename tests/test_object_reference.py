import os
import random
import uuid

import pytest

import data_index.protocols


# --- Pytest Fixtures ---
@pytest.fixture(scope="session")
def batch_size() -> int:
    return 4096


@pytest.fixture
def best_case_batch(batch_size: int) -> list[data_index.protocols.ObjectReference]:
    """Highly repetitive data, ideal for columnar/zstd compression."""
    return [
        data_index.protocols.data_index.protocols.ObjectReference(
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


# Assuming the data_index.protocols.ObjectReference class is in the same scope or imported
def test_roundtrip_fidelity(best_case_batch, worst_case_batch):
    """
    Verifies that various data profiles (high repetition vs high entropy)
    survive the full serialization/deserialization cycle with 100% fidelity.
    """
    for batch in [best_case_batch, worst_case_batch]:
        # Serialize
        b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(batch)

        print(f"compressed to {len(b64_str)}!")

        # Deserialize
        reconstructed = (
            data_index.protocols.ObjectReference.from_compressed_base64_table(b64_str)
        )

        # Verify integrity
        assert set(reconstructed) == set(batch), (
            "Data mismatch after roundtrip serialization"
        )
        assert len(reconstructed) == len(batch)


def test_single_object_roundtrip():
    """Verifies a single object instance edge case."""
    single_item = [
        data_index.protocols.ObjectReference(
            bucket="b", key="k", version_id="v1", size=100
        )
    ]

    b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(
        single_item
    )
    print(f"compressed to {len(b64_str)}!")
    reconstructed = data_index.protocols.ObjectReference.from_compressed_base64_table(
        b64_str
    )

    assert reconstructed == single_item


def test_none_value_handling():
    """Verifies that Optional/None fields are preserved correctly."""
    item_with_none = [
        data_index.protocols.ObjectReference(
            bucket="b", key="k", version_id=None, size=None
        )
    ]

    b64_str = data_index.protocols.ObjectReference.to_compressed_base64_table(
        item_with_none
    )
    print(f"compressed to {len(b64_str)}!")
    reconstructed = data_index.protocols.ObjectReference.from_compressed_base64_table(
        b64_str
    )

    assert reconstructed[0].version_id is None
    assert reconstructed[0].size is None
