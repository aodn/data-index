import dataclasses
import pathlib

import polars

from data_index.protocols import StructuredMetadata


class ParquetSink:
    """StructuredSink implementation that writes Structured Metadata to a local Parquet file."""

    def __init__(self, path: pathlib.Path = pathlib.Path("structured_metadata.parquet")) -> None:
        self.path = path

    def provision(self) -> None:
        """Create the parent directory for the Parquet file if it doesn't exist."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, data: list[StructuredMetadata]) -> None:
        rows = [dataclasses.asdict(row) for row in data]
        df = polars.DataFrame(rows, schema=StructuredMetadata.polars_schema) if rows else polars.DataFrame(schema=StructuredMetadata.polars_schema)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(self.path)
