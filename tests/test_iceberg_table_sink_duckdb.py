from unittest.mock import MagicMock

import pytest

import data_index.iceberg_config
import data_index.schema.metadata
import data_index.sink.iceberg_table_sink


def _structured_metadata(
    hash_value: str, key: str
) -> data_index.schema.metadata.StructuredMetadata:
    return data_index.schema.metadata.StructuredMetadata(
        bucket="imos-data",
        key=key,
        version_id="v1",
        hash=hash_value,
        file_format="NETCDF4",
        facility="ANMN",
    )


class _FakeDuckDBConnection:
    def __init__(self):
        self.executed: list[str] = []
        self.last_registered_table = None
        self.closed = False

    def execute(self, query: str):
        self.executed.append(query)
        return self

    def register(self, name: str, table):
        assert name == "upserts"
        self.last_registered_table = table

    def unregister(self, name: str):
        assert name == "upserts"

    def close(self):
        self.closed = True


def _table_config() -> data_index.iceberg_config.IcebergTableConfig:
    return data_index.iceberg_config.IcebergTableConfig(
        catalog_config=data_index.iceberg_config.S3TablesCatalogConfig(
            region="ap-southeast-2",
            arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
        ),
        namespace="data_index",
        table_name="structured_metadata_v5",
    )


def test_routes_write_to_pyiceberg_branch(monkeypatch):
    sink = data_index.sink.iceberg_table_sink.IcebergTableSink(
        schema_kind="structured",
        iceberg_table_config=_table_config(),
        write_engine="pyiceberg",
    )

    pyiceberg_writer = MagicMock()
    duckdb_writer = MagicMock()
    monkeypatch.setattr(sink, "_write_pyiceberg", pyiceberg_writer)
    monkeypatch.setattr(sink, "_write_duckdb", duckdb_writer)

    sink.write(metadata=[_structured_metadata(hash_value="h1", key="k1")])

    pyiceberg_writer.assert_called_once()
    duckdb_writer.assert_not_called()


def test_duckdb_write_merges_on_hash_and_dedupes_by_last_row(monkeypatch):
    fake_connection = _FakeDuckDBConnection()
    sink = data_index.sink.iceberg_table_sink.IcebergTableSink(
        schema_kind="structured",
        iceberg_table_config=_table_config(),
        write_engine="duckdb",
    )

    monkeypatch.setattr(
        data_index.iceberg_config.iceberg_table_config.IcebergTableConfig,
        "build_duckdb_connection",
        lambda self: fake_connection,
    )
    monkeypatch.setattr(
        data_index.sink.iceberg_table_sink.duckdb, "__version__", "1.5.3"
    )

    sink.write(
        metadata=[
            _structured_metadata(hash_value="dup", key="k1"),
            _structured_metadata(hash_value="dup", key="k2"),
        ]
    )

    assert fake_connection.last_registered_table is not None
    assert fake_connection.last_registered_table.num_rows == 1

    merge_statements = [
        statement
        for statement in fake_connection.executed
        if statement.startswith("MERGE INTO")
    ]
    assert len(merge_statements) == 1
    merge_sql = merge_statements[0]
    assert 'ON target."hash" = upserts."hash"' in merge_sql
    assert merge_sql.count('target."hash" = upserts."hash"') == 1
    assert "WHEN NOT MATCHED THEN INSERT BY NAME" in merge_sql
    assert fake_connection.closed is True


def test_duckdb_write_requires_minimum_version(monkeypatch):
    sink = data_index.sink.iceberg_table_sink.IcebergTableSink(
        schema_kind="structured",
        iceberg_table_config=_table_config(),
        write_engine="duckdb",
    )

    monkeypatch.setattr(
        data_index.sink.iceberg_table_sink.duckdb, "__version__", "1.4.0"
    )

    with pytest.raises(RuntimeError, match=r"duckdb>=1\.5\.3"):
        sink.write(metadata=[_structured_metadata(hash_value="h1", key="k1")])
