import json

import polars

from data_index.unstructured_sink.parquet_sink import ParquetSink


def test_provision_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.provision()

    assert path.parent.exists()


def test_writes_rows_with_identity_and_json_metadata(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)
    data = {
        "s3://bucket/a.nc?versionId=v1": {"title": "A", "count": 1},
        "s3://bucket/b.nc?versionId=v2": {"title": "B", "count": 2},
    }

    sink.write(data)

    df = polars.read_parquet(path)
    assert set(df["bucket"].to_list()) == {"bucket"}
    assert set(df["key"].to_list()) == {"a.nc", "b.nc"}
    assert set(df["version_id"].to_list()) == {"v1", "v2"}
    for row in df.iter_rows(named=True):
        parsed = json.loads(row["metadata"])
        if row["version_id"] == "v1":
            assert parsed == {"title": "A", "count": 1}
        if row["version_id"] == "v2":
            assert parsed == {"title": "B", "count": 2}


def test_writes_empty_parquet_with_correct_schema_for_empty_input(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write({})

    df = polars.read_parquet(path)
    assert "bucket" in df.columns
    assert "key" in df.columns
    assert "version_id" in df.columns
    assert "facility" in df.columns
    assert "metadata" in df.columns
    assert len(df) == 0


def test_creates_parent_directories_if_needed(tmp_path):
    path = tmp_path / "nested" / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write({"s3://bucket/x.nc?versionId=v1": {"key": "value"}})

    assert path.exists()


def test_appends_on_subsequent_writes(tmp_path):
    path = tmp_path / "out.parquet"
    sink = ParquetSink(path=path)

    sink.write({"s3://bucket/a.nc?versionId=v1": {"title": "A"}})
    sink.write({"s3://bucket/b.nc?versionId=v2": {"title": "B"}})

    df = polars.read_parquet(path)
    assert len(df) == 2
    assert set(df["key"].to_list()) == {"a.nc", "b.nc"}
