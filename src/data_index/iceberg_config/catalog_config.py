from __future__ import annotations

import typing

import pydantic
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.sql import SqlCatalog


class CatalogConfig(typing.Protocol):
    def build(self) -> Catalog: ...

    def duckdb_setup_sql(self, *, catalog_alias: str) -> list[str]: ...


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

    def duckdb_setup_sql(self, *, catalog_alias: str) -> list[str]:
        escaped_arn = self.arn.replace("'", "''")
        escaped_region = self.region.replace("'", "''")
        escaped_alias = catalog_alias.replace('"', '""')
        secret_name = f"{catalog_alias}_secret".replace('"', '""')

        return [
            "INSTALL aws",
            "INSTALL httpfs",
            "INSTALL iceberg",
            "LOAD aws",
            "LOAD httpfs",
            "LOAD iceberg",
            f"CREATE OR REPLACE SECRET \"{secret_name}\" (TYPE s3, PROVIDER credential_chain, REGION '{escaped_region}')",
            f'ATTACH \'{escaped_arn}\' AS "{escaped_alias}" (TYPE iceberg, ENDPOINT_TYPE s3_tables, SECRET "{secret_name}")',
        ]


class SqliteCatalogConfig(pydantic.BaseModel):
    """Builds a local SQLite-backed Iceberg catalog. Useful for local runs and tests."""

    name: str = "data-index"
    uri: str
    warehouse: str

    def build(self) -> Catalog:
        return SqlCatalog(self.name, uri=self.uri, warehouse=self.warehouse)

    def duckdb_setup_sql(self, *, catalog_alias: str) -> list[str]:
        raise NotImplementedError(
            "DuckDB Iceberg writes are supported only for S3TablesCatalogConfig."
        )
