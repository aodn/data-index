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


def test_provision_sets_schema_version_table_property(table_config):
    table = table_config.load()

    assert table.properties["schema_version"] == "1"


def test_writes_identity_facility_and_json_metadata(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)
    data = {
        "s3://imos-data/IMOS/ANMN/a.nc?versionId=v1": {"title": "A", "count": 1},
        "s3://imos-data/IMOS/ANMN/b.nc?versionId=v2": {"title": "B", "count": 2},
    }

    sink.write(data)

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2
    assert set(df["bucket"]) == {"imos-data"}
    assert set(df["key"]) == {"IMOS/ANMN/a.nc", "IMOS/ANMN/b.nc"}
    assert set(df["version_id"]) == {"v1", "v2"}
    for _, row in df.iterrows():
        parsed = json.loads(row["metadata"])
        if row["version_id"] == "v1":
            assert parsed == {"title": "A", "count": 1}
        if row["version_id"] == "v2":
            assert parsed == {"title": "B", "count": 2}


def test_derives_facility_as_second_path_segment(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://imos-data/IMOS/ANMN/NSW/file.nc?versionId=v1": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert df["facility"].iloc[0] == "ANMN"


def test_unknown_facility_for_short_uri(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://bucket/file.nc?versionId=v1": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert (df["facility"] == "UNKNOWN").all()


def test_unknown_facility_for_single_segment_key(table_config):
    """A file one level deep must not have its filename treated as its facility."""
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({"s3://bucket/IMOS/file.nc?versionId=v1": {"key": "val"}})

    df = table_config.load().scan().to_pandas()
    assert (df["facility"] == "UNKNOWN").all()


def test_upserts_on_subsequent_writes(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)
    uri = "s3://imos-data/IMOS/ANMN/a.nc?versionId=v1"

    sink.write({uri: {"x": 1}})
    sink.write({uri: {"x": 2}})

    df = table_config.load().scan().to_pandas()
    assert len(df) == 1
    assert json.loads(df["metadata"].iloc[0]) == {"x": 2}


def test_upsert_replaces_existing_and_inserts_new_rows(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)
    uri_a = "s3://imos-data/IMOS/ANMN/a.nc?versionId=v1"
    uri_b = "s3://imos-data/IMOS/ANMN/b.nc?versionId=v2"

    sink.write({uri_a: {"x": 1}})
    sink.write({uri_a: {"x": 2}, uri_b: {"x": 3}})

    df = table_config.load().scan().to_pandas()
    metadata_by_identity = {
        (row["bucket"], row["key"], row["version_id"]): json.loads(row["metadata"])
        for _, row in df.iterrows()
    }
    assert len(df) == 2
    assert metadata_by_identity[("imos-data", "IMOS/ANMN/a.nc", "v1")] == {"x": 2}
    assert metadata_by_identity[("imos-data", "IMOS/ANMN/b.nc", "v2")] == {"x": 3}


def test_empty_write_is_noop(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({})

    df = table_config.load().scan().to_pandas()
    assert len(df) == 0
