from __future__ import annotations

import duckdb
import pydantic
import pyiceberg.table

from .catalog_config import S3TablesCatalogConfig, SqliteCatalogConfig


class IcebergTableConfig(pydantic.BaseModel):
    """Resolves a PyIceberg table from any catalog."""

    catalog_config: S3TablesCatalogConfig | SqliteCatalogConfig
    namespace: str
    table_name: str

    def load(self) -> pyiceberg.table.Table:
        catalog = self.catalog_config.build()
        return catalog.load_table((self.namespace, self.table_name))

    @property
    def duckdb_catalog_alias(self) -> str:
        return "iceberg_catalog"

    @staticmethod
    def _quoted_identifier(value: str) -> str:
        escaped = value.replace('"', '""')
        return f'"{escaped}"'

    @property
    def duckdb_table_identifier(self) -> str:
        return ".".join(
            (
                self._quoted_identifier(self.duckdb_catalog_alias),
                self._quoted_identifier(self.namespace),
                self._quoted_identifier(self.table_name),
            )
        )

    def build_duckdb_connection(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(database=":memory:")
        try:
            setup_statements = self.catalog_config.duckdb_setup_sql(
                catalog_alias=self.duckdb_catalog_alias
            )
            for statement in setup_statements:
                connection.execute(statement)
        except Exception:
            connection.close()
            raise
        return connection
