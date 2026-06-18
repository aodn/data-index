import pathlib

import polars
import pytest

from data_index.iceberg_config import SqliteCatalogConfig
from data_index.iceberg_config.iceberg_table_config import IcebergTableConfig
from data_index.iceberg_config.table_scan_config import IcebergTableScanConfig
from data_index.inventory_source.s3_table import (
    S3TableFacilitySubsetInventorySource,
    S3TableInventorySource,
)

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
                "key": "IMOS/ACORN/a1.nc",
                "size": 1,
                "facility": "ACORN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a2.nc",
                "size": 2,
                "facility": "ACORN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a3.nc",
                "size": 3,
                "facility": "ACORN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b1.nc",
                "size": 4,
                "facility": "ANMN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b2.nc",
                "size": 5,
                "facility": "ANMN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b3.nc",
                "size": 6,
                "facility": "ANMN",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ABOS/c1.nc",
                "size": 7,
                "facility": "ABOS",
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ABOS/c2.nc",
                "size": 8,
                "facility": "ABOS",
            },
            {
                "bucket": "imos-data",
                "key": "misc/other.nc",
                "size": 9,
                "facility": None,
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
    from pyiceberg.types import LongType, NestedField, StringType

    return Schema(
        NestedField(1, "bucket", StringType(), required=True),
        NestedField(2, "key", StringType(), required=True),
        NestedField(3, "size", LongType(), required=False),
        NestedField(4, "facility", StringType(), required=False),
    )


def _write_rows(table, rows: list[dict]) -> None:
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("bucket", pa.string(), nullable=False),
            pa.field("key", pa.string(), nullable=False),
            pa.field("size", pa.int64(), nullable=True),
            pa.field("facility", pa.string(), nullable=True),
        ]
    )
    arrow_table = pa.table(
        {
            "bucket": pa.array([r["bucket"] for r in rows], type=pa.string()),
            "key": pa.array([r["key"] for r in rows], type=pa.string()),
            "size": pa.array([r["size"] for r in rows], type=pa.int64()),
            "facility": pa.array([r["facility"] for r in rows], type=pa.string()),
        },
        schema=schema,
    )
    table.append(arrow_table)


def test_inventory_filters_by_row_filter_key_prefix(table_config):
    source = S3TableInventorySource(
        table_config=table_config,
        table_scan_config=IcebergTableScanConfig(
            row_filter="key LIKE 'IMOS/ACORN/%'",
        ),
    )

    df = source.inventory()

    assert isinstance(df, polars.DataFrame)
    assert len(df) == 3
    assert all("/IMOS/ACORN/" in uri for uri in df["s3_uri"].to_list())


def test_inventory_filters_by_row_filter_facility_subset(table_config):
    source = S3TableInventorySource(
        table_config=table_config,
        table_scan_config=IcebergTableScanConfig(
            row_filter="facility IN ('ANMN', 'ABOS')",
        ),
    )

    df = source.inventory()

    assert len(df) == 5
    assert not df["s3_uri"].str.contains("/IMOS/ACORN/").any()
    assert not df["s3_uri"].str.contains("/misc/").any()


def test_inventory_subsets_per_selected_facility(table_config):
    source = S3TableFacilitySubsetInventorySource(
        table_config=table_config,
        table_scan_config=IcebergTableScanConfig(
            row_filter="facility IN ('ACORN', 'ANMN')",
        ),
        subset_per_facility=2,
    )

    df = source.inventory()

    assert len(df.filter(polars.col("s3_uri").str.contains("/IMOS/ACORN/"))) == 2
    assert len(df.filter(polars.col("s3_uri").str.contains("/IMOS/ANMN/"))) == 2
    assert not df["s3_uri"].str.contains("/IMOS/ABOS/").any()
