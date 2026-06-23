import dataclasses

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, NestedField, StringType

from data_index.iceberg_config import IcebergTableConfig, SqliteCatalogConfig
from data_index.protocols import ObjectReference
from data_index.schema.metadata import StructuredMetadata
from data_index.structured_sink.s3_table_sink import StructuredS3TableSink

NAMESPACE = "test_ns"
TABLE_NAME = "structured_metadata"


@pytest.fixture
def table_config(tmp_path):
    catalog_config = SqliteCatalogConfig(
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=str(tmp_path / "warehouse"),
    )
    config = IcebergTableConfig(
        catalog_config=catalog_config,
        namespace=NAMESPACE,
        table_name=TABLE_NAME,
    )
    StructuredS3TableSink(iceberg_table_config=config).provision()
    return config


def make_metadata(key: str, **kwargs) -> StructuredMetadata:
    object_reference = ObjectReference(
        bucket="imos-data",
        key=key,
        version_id="v1",
        size=0,
        xarray_handle=None,
    )
    defaults = dict(
        bucket=object_reference.bucket,
        key=object_reference.key,
        version_id=object_reference.version_id,
        hash=object_reference.hash,
        facility="ANMN",
        geospatial_lat_min=-10.0,
        geospatial_lat_max=10.0,
        geospatial_lon_min=100.0,
        geospatial_lon_max=110.0,
        time_coverage_start="2020-01-01",
        time_coverage_end="2020-06-01",
        crs="EPSG:4326",
        file_format="NETCDF4",
    )
    defaults.update(kwargs)
    return StructuredMetadata(**defaults)


def test_provision_is_idempotent(table_config):
    StructuredS3TableSink(
        iceberg_table_config=table_config
    ).provision()  # table already exists — must not raise


def test_writes_rows_with_correct_values(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)
    rows = [
        make_metadata(key="IMOS/ANMN/a.nc", geospatial_lat_min=-1.0),
        make_metadata(key="IMOS/ANMN/b.nc", geospatial_lat_min=-2.0),
    ]

    sink.write(rows)

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2
    assert set(df["bucket"]) == {"imos-data"}
    assert set(df["key"]) == {"IMOS/ANMN/a.nc", "IMOS/ANMN/b.nc"}
    assert set(df["schema_version"]) == {StructuredMetadata.SCHEMA_VERSION}


def test_upserts_on_subsequent_writes(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)
    key = "IMOS/ANMN/a.nc"

    sink.write([make_metadata(key=key, geospatial_lat_min=-1.0)])
    sink.write([make_metadata(key=key, geospatial_lat_min=-2.0)])

    df = table_config.load().scan().to_pandas()
    assert len(df) == 1
    assert df["geospatial_lat_min"].iloc[0] == -2.0


def test_upsert_replaces_existing_and_inserts_new_rows(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)
    key_a = "IMOS/ANMN/a.nc"
    key_b = "IMOS/ANMN/b.nc"

    sink.write([make_metadata(key=key_a, geospatial_lat_min=-1.0)])
    sink.write(
        [
            make_metadata(key=key_a, geospatial_lat_min=-2.0),
            make_metadata(key=key_b, geospatial_lat_min=-3.0),
        ]
    )

    df = table_config.load().scan().to_pandas()
    lat_by_key = dict(zip(df["key"], df["geospatial_lat_min"], strict=True))
    assert len(df) == 2
    assert lat_by_key[key_a] == -2.0
    assert lat_by_key[key_b] == -3.0


def test_empty_write_is_noop(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write([])

    df = table_config.load().scan().to_pandas()
    assert len(df) == 0


def test_null_fields_survive_roundtrip(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write(
        [
            make_metadata(
                key="a.nc",
                geospatial_lat_min=None,
                geospatial_lat_max=None,
                crs=None,
            )
        ]
    )

    df = table_config.load().scan().to_pandas()
    assert df["geospatial_lat_min"].isna().all()
    assert df["crs"].isna().all()


def test_writes_extended_structured_metadata_fields(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write(
        [
            make_metadata(
                key="a.nc",
                keywords="ocean,temp",
                instrument="CTD",
                metadata_uuid="123e4567-e89b-12d3-a456-426614174000",
            )
        ]
    )

    df = table_config.load().scan().to_pandas()
    assert df["keywords"].iloc[0] == "ocean,temp"
    assert df["instrument"].iloc[0] == "CTD"
    assert df["metadata_uuid"].iloc[0] == "123e4567-e89b-12d3-a456-426614174000"


def test_provision_evolves_existing_legacy_schema(tmp_path):
    config = IcebergTableConfig(
        catalog_config=SqliteCatalogConfig(
            uri=f"sqlite:///{tmp_path}/catalog.db",
            warehouse=str(tmp_path / "warehouse"),
        ),
        namespace=NAMESPACE,
        table_name=TABLE_NAME,
    )
    sink = StructuredS3TableSink(iceberg_table_config=config)

    sink.catalog.create_namespace(config.namespace)
    sink.catalog.create_table(
        identifier=(config.namespace, config.table_name),
        schema=Schema(
            NestedField(
                field_id=1, name="s3_uri", field_type=StringType(), required=True
            ),
            NestedField(
                field_id=2, name="lat_min", field_type=DoubleType(), required=False
            ),
            NestedField(
                field_id=3, name="lat_max", field_type=DoubleType(), required=False
            ),
            NestedField(
                field_id=4, name="lon_min", field_type=DoubleType(), required=False
            ),
            NestedField(
                field_id=5, name="lon_max", field_type=DoubleType(), required=False
            ),
            NestedField(
                field_id=6, name="time_min", field_type=StringType(), required=False
            ),
            NestedField(
                field_id=7, name="time_max", field_type=StringType(), required=False
            ),
            NestedField(
                field_id=8, name="crs", field_type=StringType(), required=False
            ),
            NestedField(
                field_id=9, name="file_format", field_type=StringType(), required=False
            ),
            NestedField(
                field_id=10, name="collection", field_type=StringType(), required=False
            ),
        ),
    )

    sink.provision(reset=True)

    schema = config.load().schema()
    assert schema.find_field("keywords").required is False
    assert schema.find_field("instrument").required is False
    expected_field_names = {
        field.name for field in dataclasses.fields(StructuredMetadata)
    }
    schema_field_names = {field.name for field in schema.fields}
    assert expected_field_names.issubset(schema_field_names)
