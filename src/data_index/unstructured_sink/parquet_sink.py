import json
import pathlib

import polars

from data_index._collection import derive_facility
from data_index.protocols import ObjectReference


class ParquetSink:
    """UnstructuredSink implementation that writes Unstructured Metadata to a local Parquet file."""

    def __init__(
        self, path: pathlib.Path = pathlib.Path("unstructured_metadata.parquet")
    ) -> None:
        self.path = path

    def provision(self) -> None:
        """Create the parent directory for the Parquet file if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: dict[str, dict]) -> None:
        schema = polars.Schema(
            {
                "bucket": polars.String,
                "key": polars.String,
                "version_id": polars.String,
                "facility": polars.String,
                "metadata": polars.String,
            }
        )
        parsed_rows = [
            (ObjectReference.from_s3_uri(uri), meta) for uri, meta in data.items()
        ]
        rows = [
            {
                "bucket": object_ref.bucket,
                "key": object_ref.key,
                "version_id": object_ref.version_id,
                "facility": derive_facility(object_ref.key),
                "metadata": json.dumps(meta),
            }
            for object_ref, meta in parsed_rows
        ]
        df = (
            polars.DataFrame(rows, schema=schema)
            if rows
            else polars.DataFrame(schema=schema)
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not rows:
                return
            existing_df = polars.read_parquet(self.path)
            df = polars.concat([existing_df, df], how="vertical")
        df.write_parquet(self.path)
