import pathlib
import tempfile

import polars
import pytest

from data_index.inventory_source.parquet import ParquetInventorySource


@pytest.fixture
def parquet_file(tmp_path: pathlib.Path) -> pathlib.Path:
    df = polars.DataFrame({
        "s3_uri": ["s3://bucket/a.nc", "s3://bucket/b.nc"],
        "size": [100, 200],
    })
    path = tmp_path / "inventory.parquet"
    df.write_parquet(path)
    return path


def test_parquet_inventory_source_returns_dataframe(parquet_file):
    source = ParquetInventorySource(path=parquet_file)
    df = source.inventory()

    assert isinstance(df, polars.DataFrame)
    assert list(df.columns) == ["s3_uri", "size"]
    assert len(df) == 2


def test_parquet_inventory_source_returns_correct_values(parquet_file):
    source = ParquetInventorySource(path=parquet_file)
    df = source.inventory()

    assert df["s3_uri"].to_list() == ["s3://bucket/a.nc", "s3://bucket/b.nc"]
    assert df["size"].to_list() == [100, 200]
