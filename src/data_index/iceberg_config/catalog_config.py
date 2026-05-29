from __future__ import annotations

import typing

import pydantic
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.sql import SqlCatalog


class CatalogConfig(typing.Protocol):
    def build(self) -> Catalog: ...


class S3TablesCatalogConfig(pydantic.BaseModel):
    """Builds an AWS S3 Tables REST catalog."""

    region: str
    arn: str = pydantic.Field(
        pattern=r"arn:aws[-a-z0-9]*:[a-z0-9]+:[-a-z0-9]*:[0-9]{12}:bucket/[a-z0-9_-]{3,63}"
    )

    @property
    def _uri(self) -> str:
        return f"https://s3tables.{self.region}.amazonaws.com/iceberg"

    def build(self) -> Catalog:

        return load_catalog(
            "s3tables_catalog",
            **{
                "type": "rest",
                "uri": self._uri,
                "warehouse": self.arn,
                "rest.sigv4-enabled": "true",
                "rest.signing-name": "s3tables",
                "rest.signing-region": self.region,
            },
        )


class SqliteCatalogConfig(pydantic.BaseModel):
    """Builds a local SQLite-backed Iceberg catalog. Useful for local runs and tests."""

    name: str = "data-index"
    uri: str
    warehouse: str

    def build(self) -> Catalog:
        return SqlCatalog(self.name, uri=self.uri, warehouse=self.warehouse)
