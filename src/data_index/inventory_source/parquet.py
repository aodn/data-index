from __future__ import annotations

import pathlib

import cloudpathlib
import polars
import pydantic


class ParquetInventorySource(pydantic.BaseModel):
    """Reads a Parquet file (local path or S3 URI) as an inventory DataFrame.

    The Parquet file must have `s3_uri` (String) and `size` (Int64) columns.
    """

    source: pathlib.Path | cloudpathlib.S3Path | str

    @property
    def _resolved_source(self) -> str:
        match self.source:
            case pathlib.Path():
                return str(self.source.resolve())
            case cloudpathlib.S3Path():
                return self.source.as_uri()
            case str():
                return self.source

    def inventory(self) -> polars.DataFrame:
        return polars.read_parquet(
            source=self._resolved_source,
            columns=["s3_uri", "size"],
        )
