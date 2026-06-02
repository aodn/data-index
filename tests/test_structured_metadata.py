import dataclasses

import polars
import pyarrow
from pyiceberg.types import DoubleType, ListType, StringType

from data_index.structured_metadata import StructuredMetadata


@dataclasses.dataclass
class StructuredMetadataWithList(StructuredMetadata):
    tags: list[str] | None = None


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


def test_as_polars_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_polars_schema()

    assert schema["tags"] == polars.List(polars.String)


def test_as_pyarrow_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_pyarrow_schema()

    assert schema.field("tags").type == pyarrow.list_(pyarrow.string())
    assert schema.field("tags").nullable is True


def test_as_pyiceberg_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_pyiceberg_schema()
    tags_field_type = schema.find_field("tags").field_type

    assert isinstance(tags_field_type, ListType)
    assert tags_field_type.element_type == StringType()
    assert tags_field_type.element_id > 0
    assert schema.find_field("tags").required is False
