import dataclasses

import polars
import pyarrow
from pyiceberg.types import DoubleType, StringType

from data_index.structured_metadata import StructuredMetadata


def test_as_polars_schema_maps_required_and_optional_types():
    schema = StructuredMetadata.as_polars_schema()

    assert schema["s3_uri"] == polars.String
    assert schema["lat_min"] == polars.Float64
    assert schema["time_min"] == polars.String
    assert len(schema) == len(dataclasses.fields(StructuredMetadata))


def test_as_pyarrow_schema_preserves_nullable_fields():
    schema = StructuredMetadata.as_pyarrow_schema()

    assert schema.field("s3_uri").type == pyarrow.string()
    assert schema.field("s3_uri").nullable is False
    assert schema.field("lat_min").type == pyarrow.float64()
    assert schema.field("lat_min").nullable is True


def test_as_pyiceberg_schema_preserves_nullable_fields():
    schema = StructuredMetadata.as_pyiceberg_schema()

    assert schema.find_field("s3_uri").field_type == StringType()
    assert schema.find_field("s3_uri").required is True
    assert schema.find_field("lat_min").field_type == DoubleType()
    assert schema.find_field("lat_min").required is False
    assert len(schema.fields) == len(dataclasses.fields(StructuredMetadata))
