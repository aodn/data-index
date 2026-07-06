from unittest.mock import MagicMock, call

import pytest

import data_index.iceberg_config.catalog_config
import data_index.iceberg_config.iceberg_table_config


def test_s3tables_catalog_config_emits_duckdb_setup_sql():
    config = data_index.iceberg_config.catalog_config.S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
    )

    setup_sql = config.duckdb_setup_sql(catalog_alias="my_catalog")
    setup_sql_text = "\n".join(setup_sql).lower()

    assert "install aws" in setup_sql_text
    assert "load iceberg" in setup_sql_text
    assert "provider credential_chain" in setup_sql_text
    assert "endpoint_type s3_tables" in setup_sql_text
    assert (
        "arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index"
        in setup_sql_text
    )


def test_sqlite_catalog_config_rejects_duckdb_setup_sql():
    config = data_index.iceberg_config.catalog_config.SqliteCatalogConfig(
        uri="sqlite:///tmp/catalog.db",
        warehouse="/tmp/warehouse",
    )

    with pytest.raises(NotImplementedError):
        config.duckdb_setup_sql(catalog_alias="local_catalog")


def test_iceberg_table_config_builds_duckdb_connection_with_catalog_setup(monkeypatch):
    fake_connection = MagicMock()
    fake_catalog_config = MagicMock()
    fake_catalog_config.duckdb_setup_sql.return_value = [
        "LOAD iceberg",
        "ATTACH 'warehouse' AS catalog (TYPE iceberg, ENDPOINT_TYPE s3_tables)",
    ]

    table_config = data_index.iceberg_config.iceberg_table_config.IcebergTableConfig.model_construct(
        catalog_config=fake_catalog_config,
        namespace="data_index",
        table_name="structured_metadata_v5",
    )

    connect = MagicMock(return_value=fake_connection)
    monkeypatch.setattr(
        data_index.iceberg_config.iceberg_table_config.duckdb, "connect", connect
    )

    connection = table_config.build_duckdb_connection()

    assert connection is fake_connection
    assert (
        table_config.duckdb_table_identifier
        == '"iceberg_catalog"."data_index"."structured_metadata_v5"'
    )
    fake_catalog_config.duckdb_setup_sql.assert_called_once_with(
        catalog_alias="iceberg_catalog"
    )
    assert fake_connection.execute.call_args_list == [
        call("LOAD iceberg"),
        call("ATTACH 'warehouse' AS catalog (TYPE iceberg, ENDPOINT_TYPE s3_tables)"),
    ]
