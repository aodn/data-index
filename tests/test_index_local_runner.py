import pathlib

import data_index.iceberg_config
from data_index.file_fetcher import LocalFetcher
from data_index.inventory_source import LocalGlobInventorySource
from data_index.runners import index_local


def test_local_runner_uses_local_inventory_and_fetcher_defaults():
    assert isinstance(index_local.INVENTORY_SOURCE, LocalGlobInventorySource)
    assert isinstance(index_local.FILE_FETCHER, LocalFetcher)
    assert (
        index_local.INVENTORY_SOURCE.local_version_id
        == index_local.FILE_FETCHER.local_version_id
    )


def test_local_runner_uses_sqlite_catalog_for_all_sinks():
    for sink in (
        index_local.STRUCTURED_TABLE_SINK,
        index_local.UNSTRUCTURED_TABLE_SINK,
        index_local.DEAD_LETTER_TABLE_SINK,
    ):
        assert isinstance(
            sink.iceberg_table_config.catalog_config,
            data_index.iceberg_config.SqliteCatalogConfig,
        )


def test_local_runner_warehouse_path_is_relative_local_load_dir():
    assert index_local.LOCAL_WAREHOUSE == pathlib.Path(".load/orchestrate-local")
