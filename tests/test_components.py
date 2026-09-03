import pytest

import data_index.protocols
import data_index.schema.metadata


@pytest.mark.parametrize(
    "dataclass_cls",
    [
        data_index.protocols.DeadLetter,
        data_index.schema.metadata.StructuredMetadata,
        data_index.schema.metadata.UnstructuredMetadata,
    ],
    ids=[
        "Dead Letter",
        "Structured Metadata",
        "Unstructured Metadata",
    ],
)
def test_core_schemas_contract(dataclass_cls, data_regression):
    """Ensure all core production schemas match their historical snapshot contracts.

    This contract test acts as a schema regression guard. It compiles the target
    dataclass into PyArrow, Polars, and PyIceberg representations and compares
    the serialized output against a ground-truth snapshot file on disk.

    If a schema change is intentional (e.g., adding a new field), update the
    historical snapshots by deleting them and running tests to regen files

    :param dataclass_cls: The production dataclass type being evaluated.
    :param data_regression: Pytest fixture handling snapshot storage and diffing.
    """

    # Compile schemas for your target frameworks
    arrow_schema = dataclass_cls.as_pyarrow_schema()
    polars_schema = dataclass_cls.as_polars_schema()
    iceberg_schema = dataclass_cls.as_pyiceberg_schema()

    # Serialize them into a single comprehensive contract dictionary
    contract_payload = {
        "pyarrow": {field.name: str(field.type) for field in arrow_schema},
        "polars": {name: str(dtype) for name, dtype in polars_schema.items()},
        "pyiceberg": {
            field.name: f"id={field.field_id}, type={field.field_type!s}, req={field.required}"
            for field in iceberg_schema.fields
        },
    }

    # Check against disk snapshot.
    # data_regression automatically appends the `id` (e.g., [Dead Letter]) to the filename
    data_regression.check(contract_payload)
