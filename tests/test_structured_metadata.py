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

    assert schema["bucket"] == polars.String
    assert schema["key"] == polars.String
    assert schema["version_id"] == polars.String
    assert schema["schema_version"] == polars.Int64
    assert schema["geospatial_lat_min"] == polars.Float64
    assert schema["time_coverage_start"] == polars.String
    assert len(schema) == len(dataclasses.fields(StructuredMetadata))


def test_as_pyarrow_schema_preserves_nullable_fields():
    schema = StructuredMetadata.as_pyarrow_schema()

    assert schema.field("bucket").type == pyarrow.string()
    assert schema.field("bucket").nullable is False
    assert schema.field("key").nullable is False
    assert schema.field("version_id").nullable is False
    assert schema.field("schema_version").type == pyarrow.int64()
    assert schema.field("schema_version").nullable is True
    assert schema.field("geospatial_lat_min").type == pyarrow.float64()
    assert schema.field("geospatial_lat_min").nullable is True


def test_as_pyiceberg_schema_preserves_nullable_fields():
    schema = StructuredMetadata.as_pyiceberg_schema()

    assert schema.find_field("bucket").field_type == StringType()
    assert schema.find_field("bucket").required is True
    assert schema.find_field("key").required is True
    assert schema.find_field("version_id").required is True
    assert schema.find_field("schema_version").required is False
    assert schema.find_field("geospatial_lat_min").field_type == DoubleType()
    assert schema.find_field("geospatial_lat_min").required is False
    assert len(schema.fields) == len(dataclasses.fields(StructuredMetadata))


def test_as_polars_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_polars_schema()

    assert schema["tags"] == polars.List(polars.String)


def test_as_pyarrow_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_pyarrow_schema()
    tags_type = schema.field("tags").type

    assert tags_type == pyarrow.list_(
        pyarrow.field("item", pyarrow.string(), nullable=False)
    )
    assert tags_type.value_field.nullable is False
    assert schema.field("tags").nullable is True


def test_as_pyiceberg_schema_maps_optional_list_string():
    schema = StructuredMetadataWithList.as_pyiceberg_schema()
    tags_field_type = schema.find_field("tags").field_type

    assert isinstance(tags_field_type, ListType)
    assert tags_field_type.element_type == StringType()
    assert tags_field_type.element_id > 0
    assert schema.find_field("tags").required is False


def test_structured_metadata_field_contract_updates():
    field_names = [field.name for field in dataclasses.fields(StructuredMetadata)]

    assert "s3_uri" not in field_names
    assert "file_version_quality_control" not in field_names
    assert "feature_type" in field_names
    assert "instrument_serial_number" in field_names
    assert "dimensions" in field_names
    assert "variables" in field_names
    assert "standard_names" in field_names
    assert "facility" in field_names
    assert "collection" not in field_names


def test_schema_version_is_class_var_and_not_dataclass_field():
    field_names = [field.name for field in dataclasses.fields(StructuredMetadata)]

    assert StructuredMetadata.SCHEMA_VERSION == 2
    assert "SCHEMA_VERSION" not in field_names


def test_schema_version_field_defaults_to_class_var_value():
    row = StructuredMetadata(bucket="bucket", key="file.nc", version_id="v1")

    assert row.schema_version == StructuredMetadata.SCHEMA_VERSION
