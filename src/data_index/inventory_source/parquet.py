from __future__ import annotations

import pathlib
import typing

import cloudpathlib
import polars
import pydantic

from data_index.inventory_source._contract import enforce_inventory_contract


class ParquetInventorySource(pydantic.BaseModel):
    """Reads a Parquet file (local path or S3 URI) as an inventory DataFrame.

    The Parquet file must have `bucket`, `key`, `version_id`, and `size` columns.
    """

    type: typing.Literal["parquet"] = pydantic.Field(default="parquet")

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
        return enforce_inventory_contract(
            polars.read_parquet(
                source=self._resolved_source,
                columns=["bucket", "key", "version_id", "size"],
            )
        )
