import pathlib

import cloudpathlib
import polars
import pytest

from data_index.inventory_source.parquet import ParquetInventorySource


@pytest.fixture
def parquet_file(tmp_path: pathlib.Path) -> pathlib.Path:
    df = polars.DataFrame(
        {
            "bucket": ["bucket", "bucket"],
            "key": ["a.nc", "b.nc"],
            "version_id": ["v1", "v2"],
            "size": [100, 200],
        }
    )
    path = tmp_path / "inventory.parquet"
    df.write_parquet(path)
    return path


def test_parquet_inventory_source_returns_dataframe(parquet_file):
    source = ParquetInventorySource(source=parquet_file)
    df = source.inventory()

    assert isinstance(df, polars.DataFrame)
    assert list(df.columns) == ["bucket", "key", "version_id", "size"]
    assert len(df) == 2


def test_parquet_inventory_source_returns_correct_values(parquet_file):
    source = ParquetInventorySource(source=parquet_file)
    df = source.inventory()

    assert df["bucket"].to_list() == ["bucket", "bucket"]
    assert df["key"].to_list() == ["a.nc", "b.nc"]
    assert df["version_id"].to_list() == ["v1", "v2"]
    assert df["size"].to_list() == [100, 200]


def test_parquet_inventory_source_resolves_source_correctly(parquet_file):
    path_source = ParquetInventorySource(source=parquet_file)
    assert path_source._resolved_source == str(parquet_file.resolve())

    str_source = ParquetInventorySource(
        source=cloudpathlib.S3Path(cloud_path="s3://bucket/inventory.parquet")
    )
    assert str_source._resolved_source == "s3://bucket/inventory.parquet"
