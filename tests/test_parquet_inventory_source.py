import pathlib

import cloudpathlib
import polars
import pytest

from data_index.inventory_source.parquet import ParquetInventorySource


@pytest.fixture
def parquet_file(tmp_path: pathlib.Path) -> pathlib.Path:
    df = polars.DataFrame(
        {
            "s3_uri": ["s3://bucket/a.nc", "s3://bucket/b.nc"],
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
    assert list(df.columns) == ["s3_uri", "size"]
    assert len(df) == 2


def test_parquet_inventory_source_returns_correct_values(parquet_file):
    source = ParquetInventorySource(source=parquet_file)
    df = source.inventory()

    assert df["s3_uri"].to_list() == ["s3://bucket/a.nc", "s3://bucket/b.nc"]
    assert df["size"].to_list() == [100, 200]


def test_parquet_inventory_source_resolves_source_correctly(parquet_file):
    path_source = ParquetInventorySource(source=parquet_file)
    assert path_source._resolved_source == str(parquet_file.resolve())

    str_source = ParquetInventorySource(
        source=cloudpathlib.S3Path(cloud_path="s3://bucket/inventory.parquet")
    )
    assert str_source._resolved_source == "s3://bucket/inventory.parquet"
