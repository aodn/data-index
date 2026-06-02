import dataclasses

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, NestedField, StringType

from data_index.iceberg_config import IcebergTableConfig, SqliteCatalogConfig
from data_index.structured_metadata import StructuredMetadata
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


def make_metadata(**kwargs) -> StructuredMetadata:
    defaults = dict(
        s3_uri="s3://imos-data/IMOS/ANMN/NSW/file.nc",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=100.0,
        lon_max=110.0,
        time_min="2020-01-01",
        time_max="2020-06-01",
        crs="EPSG:4326",
        file_format="NETCDF4",
        collection="ANMN",
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
        make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/a.nc", lat_min=-1.0),
        make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/b.nc", lat_min=-2.0),
    ]

    sink.write(rows)

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2
    assert set(df["s3_uri"]) == {
        "s3://imos-data/IMOS/ANMN/a.nc",
        "s3://imos-data/IMOS/ANMN/b.nc",
    }


def test_appends_on_subsequent_writes(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write([make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/a.nc")])
    sink.write([make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/b.nc")])

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2


def test_empty_write_is_noop(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write([])

    df = table_config.load().scan().to_pandas()
    assert len(df) == 0


def test_null_fields_survive_roundtrip(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write([make_metadata(lat_min=None, lat_max=None, crs=None, collection=None)])

    df = table_config.load().scan().to_pandas()
    assert df["lat_min"].isna().all()
    assert df["crs"].isna().all()
    assert df["collection"].isna().all()


def test_writes_extended_structured_metadata_fields(table_config):
    sink = StructuredS3TableSink(iceberg_table_config=table_config)

    sink.write(
        [
            make_metadata(
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

    sink.provision()

    schema = config.load().schema()
    assert schema.find_field("keywords").required is False
    assert schema.find_field("instrument").required is False
    assert len(schema.fields) == len(dataclasses.fields(StructuredMetadata))
