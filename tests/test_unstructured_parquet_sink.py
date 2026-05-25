import json

import polars

from data_index.unstructured_sink.parquet_sink import ParquetSink


def test_provision_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.provision()

    assert path.parent.exists()


def test_writes_rows_with_s3_uri_and_json_metadata(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)
    data = {
        "s3://bucket/a.nc": {"title": "A", "count": 1},
        "s3://bucket/b.nc": {"title": "B", "count": 2},
    }

    sink.write(data)

    df = polars.read_parquet(path)
    assert set(df["s3_uri"].to_list()) == set(data.keys())
    for row in df.iter_rows(named=True):
        parsed = json.loads(row["metadata"])
        assert parsed == data[row["s3_uri"]]


def test_writes_empty_parquet_with_correct_schema_for_empty_input(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write({})

    df = polars.read_parquet(path)
    assert "s3_uri" in df.columns
    assert "metadata" in df.columns
    assert len(df) == 0


def test_creates_parent_directories_if_needed(tmp_path):
    path = tmp_path / "nested" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write({"s3://bucket/x.nc": {"key": "value"}})

    assert path.exists()
