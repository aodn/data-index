import json
import pathlib

import polars


class ParquetSink:
    """UnstructuredSink implementation that writes Unstructured Metadata to a local Parquet file."""

    def __init__(self, path: pathlib.Path = pathlib.Path("unstructured_metadata.parquet")) -> None:
        self.path = path

    def write(self, data: dict[str, dict]) -> None:
        schema = polars.Schema({"s3_uri": polars.String, "metadata": polars.String})
        rows = [{"s3_uri": uri, "metadata": json.dumps(meta)} for uri, meta in data.items()]
        df = polars.DataFrame(rows, schema=schema) if rows else polars.DataFrame(schema=schema)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.path)
