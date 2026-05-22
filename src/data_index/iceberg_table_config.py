from __future__ import annotations

import pydantic
import pyiceberg.table

from data_index.catalog_config import S3TablesCatalogConfig, SqliteCatalogConfig


class IcebergTableConfig(pydantic.BaseModel):
    """Resolves a PyIceberg table from any catalog."""

    catalog_config: S3TablesCatalogConfig | SqliteCatalogConfig
    namespace: str
    table_name: str

    def load(self) -> pyiceberg.table.Table:
        catalog = self.catalog_config.build()
        return catalog.load_table((self.namespace, self.table_name))
