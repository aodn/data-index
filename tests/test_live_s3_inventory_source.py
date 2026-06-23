import pathlib

import polars
import pytest

from data_index.iceberg_config import SqliteCatalogConfig
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.inventory_source.live_s3 import LiveS3InventorySource
from data_index.s3_metadata.extract import TableScanConfig

NAMESPACE = "test-namespace"
TABLE_NAME = "inventory"


@pytest.fixture
def table_config(tmp_path: pathlib.Path) -> IcebergTableConfig:
    catalog_config = SqliteCatalogConfig(
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=str(tmp_path),
    )
    catalog = catalog_config.build()
    catalog.create_namespace(NAMESPACE)
    table = catalog.create_table(
        identifier=(NAMESPACE, TABLE_NAME),
        schema=_iceberg_schema(),
    )
    _write_rows(
        table,
        [
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/file.nc",
                "size": 1024,
                "sequence_number": "1",
                "version_id": "v1",
                "is_delete_marker": False,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/other.nc",
                "size": 2048,
                "sequence_number": "1",
                "version_id": "v2",
                "is_delete_marker": False,
            },
        ],
    )
    return IcebergTableConfig(
        catalog_config=catalog_config,
        namespace=NAMESPACE,
        table_name=TABLE_NAME,
    )


def _iceberg_schema():
    from pyiceberg.schema import Schema
    from pyiceberg.types import BooleanType, LongType, NestedField, StringType

    return Schema(
        NestedField(1, "bucket", StringType(), required=True),
        NestedField(2, "key", StringType(), required=True),
        NestedField(3, "sequence_number", StringType(), required=True),
        NestedField(4, "version_id", StringType(), required=False),
        NestedField(5, "is_delete_marker", BooleanType(), required=False),
        NestedField(6, "size", LongType(), required=False),
    )


def _write_rows(table, rows: list[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("bucket", pa.string(), nullable=False),
            pa.field("key", pa.string(), nullable=False),
            pa.field("sequence_number", pa.string(), nullable=False),
            pa.field("version_id", pa.string(), nullable=True),
            pa.field("is_delete_marker", pa.bool_(), nullable=True),
            pa.field("size", pa.int64(), nullable=True),
        ]
    )
    arrow_table = pa.table(
        {
            "bucket": pa.array([r["bucket"] for r in rows], type=pa.string()),
            "key": pa.array([r["key"] for r in rows], type=pa.string()),
            "sequence_number": pa.array(
                [r["sequence_number"] for r in rows], type=pa.string()
            ),
            "version_id": pa.array([r["version_id"] for r in rows], type=pa.string()),
            "is_delete_marker": pa.array(
                [r["is_delete_marker"] for r in rows], type=pa.bool_()
            ),
            "size": pa.array([r["size"] for r in rows], type=pa.int64()),
        },
        schema=schema,
    )
    table.append(arrow_table)


def test_inventory_returns_identity_and_size_columns(table_config, tmp_path):
    source = LiveS3InventorySource(
        table_config=table_config,
        table_scan_config=TableScanConfig(),
        path=tmp_path / "s3_metadata",
    )
    df = source.inventory()

    assert isinstance(df, polars.DataFrame)
    assert set(df.columns) == {"bucket", "key", "version_id", "size"}


def test_inventory_returns_bucket_key_and_version_values(table_config, tmp_path):
    source = LiveS3InventorySource(
        table_config=table_config,
        table_scan_config=TableScanConfig(),
        path=tmp_path / "s3_metadata",
    )
    df = source.inventory()

    assert ("imos-data", "IMOS/ACORN/file.nc", "v1") in set(
        zip(df["bucket"], df["key"], df["version_id"], strict=True)
    )
    assert ("imos-data", "IMOS/ANMN/other.nc", "v2") in set(
        zip(df["bucket"], df["key"], df["version_id"], strict=True)
    )


def test_skip_if_exists_skips_etl_when_data_present(table_config, tmp_path):
    path = tmp_path / "s3_metadata"
    source = LiveS3InventorySource(
        table_config=table_config,
        table_scan_config=TableScanConfig(),
        path=path,
        skip_if_exists=True,
    )
    # First call materialises
    df1 = source.inventory()
    # Poison the table so a re-run would fail
    table_config.catalog_config = SqliteCatalogConfig(
        uri="sqlite:////nonexistent/catalog.db",
        warehouse="/nonexistent",
    )
    # Second call should use cached data, not hit the catalog
    df2 = source.inventory()

    assert df1.equals(df2)


def test_skip_if_exists_false_reruns_etl(table_config, tmp_path):
    path = tmp_path / "s3_metadata"
    source = LiveS3InventorySource(
        table_config=table_config,
        table_scan_config=TableScanConfig(),
        path=path,
        skip_if_exists=False,
    )
    df1 = source.inventory()

    # Add a new row to the table
    table = table_config.load()
    _write_rows(
        table,
        [
            {
                "bucket": "imos-data",
                "key": "IMOS/NEW/new.nc",
                "size": 512,
                "sequence_number": "1",
                "version_id": "v3",
                "is_delete_marker": False,
            },
        ],
    )

    df2 = source.inventory()

    assert len(df2) == len(df1) + 1
