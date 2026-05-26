import json

import pytest

from data_index.iceberg_config import IcebergTableConfig, SqliteCatalogConfig
from data_index.unstructured_sink.s3_table_sink import UnstructuredS3TableSink

NAMESPACE = "test_ns"
TABLE_NAME = "unstructured_metadata"


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
    UnstructuredS3TableSink(iceberg_table_config=config).provision()
    return config


def test_provision_is_idempotent(table_config):
    UnstructuredS3TableSink(
        iceberg_table_config=table_config
    ).provision()  # table already exists — must not raise


def test_writes_s3_uri_collection_and_json_metadata(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)
    data = {
        "s3://imos-data/IMOS/ANMN/a.nc": {"title": "A", "count": 1},
        "s3://imos-data/IMOS/ANMN/b.nc": {"title": "B", "count": 2},
    }

    sink.write(data)

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2
    assert set(df["s3_uri"]) == set(data.keys())
    for _, row in df.iterrows():
        parsed = json.loads(row["metadata"])
        assert parsed == data[row["s3_uri"]]


def test_derives_collection_as_second_path_segment(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://imos-data/IMOS/ANMN/NSW/file.nc": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert df["collection"].iloc[0] == "ANMN"


def test_null_collection_for_short_uri(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://bucket/file.nc": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert df["collection"].isna().all()


def test_null_collection_for_single_segment_key(table_config):
    """A file one level deep must not have its filename treated as its collection."""
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://bucket/IMOS/file.nc": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert df["collection"].isna().all()


def test_appends_on_subsequent_writes(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://imos-data/IMOS/ANMN/a.nc": {"x": 1}})
    sink.write({"s3://imos-data/IMOS/ANMN/b.nc": {"x": 2}})

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2


def test_empty_write_is_noop(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({})

    df = table_config.load().scan().to_pandas()
    assert len(df) == 0
