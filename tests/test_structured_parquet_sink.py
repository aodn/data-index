import pathlib

import polars
import pytest

from data_index.protocols import StructuredMetadata
from data_index.structured_sink.parquet_sink import ParquetSink


def make_metadata(**kwargs) -> StructuredMetadata:
    defaults = dict(
        s3_uri="s3://bucket/file.nc",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=100.0,
        lon_max=110.0,
        time_min="2020-01-01",
        time_max="2020-06-01",
        crs="EPSG:4326",
        file_format=None,
    )
    defaults.update(kwargs)
    return StructuredMetadata(**defaults)


def test_writes_rows_with_correct_schema_and_values(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)
    rows = [
        make_metadata(s3_uri="s3://bucket/a.nc", lat_min=-1.0, lat_max=1.0),
        make_metadata(s3_uri="s3://bucket/b.nc", lat_min=-2.0, lat_max=2.0),
    ]

    sink.write(rows)

    df = polars.read_parquet(path)
    assert df.schema == StructuredMetadata.polars_schema
    assert len(df) == 2
    assert df["s3_uri"].to_list() == ["s3://bucket/a.nc", "s3://bucket/b.nc"]
    assert df["lat_min"].to_list() == pytest.approx([-1.0, -2.0])


def test_writes_empty_parquet_with_correct_schema_for_empty_input(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([])

    df = polars.read_parquet(path)
    assert df.schema == StructuredMetadata.polars_schema
    assert len(df) == 0


def test_creates_parent_directories_if_needed(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([make_metadata()])

    assert path.exists()


def test_writes_none_fields_as_null(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([make_metadata(lat_min=None, lat_max=None, crs=None)])

    df = polars.read_parquet(path)
    assert df["lat_min"][0] is None
    assert df["crs"][0] is None
