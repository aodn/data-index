import decimal

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

    assert row.schema_version == UnstructuredMetadata.SCHEMA_VERSION


def test_structured_as_dynamodb_item_converts_floats_and_maps():
    row = StructuredMetadata(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        hash="hash",
        file_format="NETCDF4",
        facility="ANMN",
        geospatial_lat_min=12.5,
        dimension_sizes={"time": 10},
    )

    item = row.as_dynamodb_item()

    assert item["geospatial_lat_min"] == decimal.Decimal("12.5")
    assert item["dimension_sizes"] == {"time": 10}


def test_structured_as_dynamodb_item_coerces_non_finite_float_to_none():
    row = StructuredMetadata(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        hash="hash",
        file_format="NETCDF4",
        facility="ANMN",
        geospatial_lat_min=float("nan"),
    )

    item = row.as_dynamodb_item()

    assert "geospatial_lat_min" not in item


def test_structured_as_dynamodb_item_omits_nulls_when_requested():
    row = StructuredMetadata(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        hash="hash",
        file_format="NETCDF4",
        facility="ANMN",
        geospatial_lat_min=None,
    )

    item = row.as_dynamodb_item(include_nulls=False)

    assert "geospatial_lat_min" not in item
