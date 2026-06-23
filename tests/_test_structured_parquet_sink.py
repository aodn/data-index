import polars
import pytest

from data_index.structured_metadata import StructuredMetadata
from data_index.structured_sink.parquet_sink import ParquetSink


def make_metadata(**kwargs) -> StructuredMetadata:
    defaults = dict(
        bucket="bucket",
        key="file.nc",
        version_id="v1",
        facility="UNKNOWN",
        geospatial_lat_min=-10.0,
        geospatial_lat_max=10.0,
        geospatial_lon_min=100.0,
        geospatial_lon_max=110.0,
        time_coverage_start="2020-01-01",
        time_coverage_end="2020-06-01",
        crs="EPSG:4326",
        file_format=None,
    )
    defaults.update(kwargs)
    return StructuredMetadata(**defaults)


def test_provision_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.provision()

    assert path.parent.exists()


def test_writes_rows_with_correct_schema_and_values(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)
    rows = [
        make_metadata(
            key="a.nc",
            geospatial_lat_min=-1.0,
            geospatial_lat_max=1.0,
        ),
        make_metadata(
            key="b.nc",
            geospatial_lat_min=-2.0,
            geospatial_lat_max=2.0,
        ),
    ]

    sink.write(rows)

    df = polars.read_parquet(path)
    assert df.schema == StructuredMetadata.as_polars_schema()
    assert len(df) == 2
    assert df["bucket"].to_list() == ["bucket", "bucket"]
    assert df["key"].to_list() == ["a.nc", "b.nc"]
    assert df["version_id"].to_list() == ["v1", "v1"]
    assert df["schema_version"].to_list() == [
        StructuredMetadata.SCHEMA_VERSION,
        StructuredMetadata.SCHEMA_VERSION,
    ]
    assert df["geospatial_lat_min"].to_list() == pytest.approx([-1.0, -2.0])


def test_writes_empty_parquet_with_correct_schema_for_empty_input(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([])

    df = polars.read_parquet(path)
    assert df.schema == StructuredMetadata.as_polars_schema()
    assert len(df) == 0


def test_creates_parent_directories_if_needed(tmp_path):
    path = tmp_path / "nested" / "dir" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([make_metadata()])

    assert path.exists()


def test_writes_none_fields_as_null(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write(
        [make_metadata(geospatial_lat_min=None, geospatial_lat_max=None, crs=None)]
    )

    df = polars.read_parquet(path)
    assert df["geospatial_lat_min"][0] is None
    assert df["crs"][0] is None


def test_appends_on_subsequent_writes(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write([make_metadata(key="a.nc", geospatial_lat_min=-1.0)])
    sink.write([make_metadata(key="b.nc", geospatial_lat_min=-2.0)])

    df = polars.read_parquet(path)
    assert len(df) == 2
    assert set(df["key"].to_list()) == {"a.nc", "b.nc"}
