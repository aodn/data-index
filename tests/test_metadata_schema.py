import pytest

from data_index.protocols import ObjectReference
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata


@pytest.fixture(scope="session")
def object_reference():
    """Provides a valid ObjectReference instance for metadata tests."""
    return ObjectReference(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        size=0,
        xarray_handle=None,
    )


def test_structured_schema_version_field_defaults_to_class_var_value(object_reference):
    row = StructuredMetadata(
        bucket=object_reference.bucket,
        key=object_reference.key,
        version_id=object_reference.version_id,
        hash=object_reference.hash,
        file_format="",
        facility="",
    )

    assert row.schema_version == StructuredMetadata.SCHEMA_VERSION


def test_unstructured_schema_version_field_defaults_to_class_var_value(
    object_reference,
):
    row = UnstructuredMetadata(
        bucket=object_reference.bucket,
        key=object_reference.key,
        version_id=object_reference.version_id,
        hash=object_reference.hash,
        metadata="",
        file_format="",
        facility="",
    )

    assert row.schema_version == StructuredMetadata.SCHEMA_VERSION
