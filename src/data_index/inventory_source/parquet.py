from __future__ import annotations

import os

import polars


class ParquetInventorySource:
    """Reads a Parquet file (local path or S3 URI) as an inventory DataFrame.

    The Parquet file must have `s3_uri` (String) and `size` (Int64) columns.
    """

    def __init__(self, path: os.PathLike | str) -> None:
        self._path = path

    def inventory(self) -> polars.DataFrame:
        return polars.read_parquet(self._path, columns=["s3_uri", "size"])
