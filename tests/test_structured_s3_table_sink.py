import pytest
from pyiceberg.catalog.sql import SqlCatalog

from data_index.protocols import StructuredMetadata
from data_index.structured_sink.s3_table_sink import StructuredS3TableSink

NAMESPACE = "test_ns"
TABLE_NAME = "structured_metadata"


@pytest.fixture
def catalog(tmp_path):
    cat = SqlCatalog(
        "test",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=str(tmp_path / "warehouse"),
    )
    sink = StructuredS3TableSink(cat, NAMESPACE, TABLE_NAME)
    sink.provision()
    return cat


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


def test_provision_is_idempotent(catalog):
    sink = StructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)
    sink.provision()  # table already exists — must not raise


def test_writes_rows_with_correct_values(catalog):
    sink = StructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)
    rows = [
        make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/a.nc", lat_min=-1.0),
        make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/b.nc", lat_min=-2.0),
    ]

    sink.write(rows)

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 2
    assert set(df["s3_uri"]) == {
        "s3://imos-data/IMOS/ANMN/a.nc",
        "s3://imos-data/IMOS/ANMN/b.nc",
    }


def test_appends_on_subsequent_writes(catalog):
    sink = StructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write([make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/a.nc")])
    sink.write([make_metadata(s3_uri="s3://imos-data/IMOS/ANMN/b.nc")])

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 2


def test_empty_write_is_noop(catalog):
    sink = StructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write([])

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 0


def test_null_fields_survive_roundtrip(catalog):
    sink = StructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write([make_metadata(lat_min=None, lat_max=None, crs=None, collection=None)])

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert df["lat_min"].isna().all()
    assert df["crs"].isna().all()
    assert df["collection"].isna().all()
