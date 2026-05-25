import json

import pytest
from pyiceberg.catalog.sql import SqlCatalog

from data_index.unstructured_sink.s3_table_sink import UnstructuredS3TableSink

NAMESPACE = "test_ns"
TABLE_NAME = "unstructured_metadata"


@pytest.fixture
def catalog(tmp_path):
    cat = SqlCatalog(
        "test",
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=str(tmp_path / "warehouse"),
    )
    sink = UnstructuredS3TableSink(cat, NAMESPACE, TABLE_NAME)
    sink.provision()
    return cat


def test_provision_is_idempotent(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)
    sink.provision()  # table already exists — must not raise


def test_writes_s3_uri_collection_and_json_metadata(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)
    data = {
        "s3://imos-data/IMOS/ANMN/a.nc": {"title": "A", "count": 1},
        "s3://imos-data/IMOS/ANMN/b.nc": {"title": "B", "count": 2},
    }

    sink.write(data)

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 2
    assert set(df["s3_uri"]) == set(data.keys())
    for _, row in df.iterrows():
        parsed = json.loads(row["metadata"])
        assert parsed == data[row["s3_uri"]]


def test_derives_collection_as_second_path_segment(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write({"s3://imos-data/IMOS/ANMN/NSW/file.nc": {"key": "val"}})

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert df["collection"].iloc[0] == "ANMN"


def test_null_collection_for_short_uri(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write({"s3://bucket/file.nc": {"key": "val"}})

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert df["collection"].isna().all()


def test_null_collection_for_single_segment_key(catalog):
    """A file one level deep must not have its filename treated as its collection."""
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write({"s3://bucket/IMOS/file.nc": {"key": "val"}})

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert df["collection"].isna().all()


def test_appends_on_subsequent_writes(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write({"s3://imos-data/IMOS/ANMN/a.nc": {"x": 1}})
    sink.write({"s3://imos-data/IMOS/ANMN/b.nc": {"x": 2}})

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 2


def test_empty_write_is_noop(catalog):
    sink = UnstructuredS3TableSink(catalog, NAMESPACE, TABLE_NAME)

    sink.write({})

    table = catalog.load_table((NAMESPACE, TABLE_NAME))
    df = table.scan().to_pandas()
    assert len(df) == 0
