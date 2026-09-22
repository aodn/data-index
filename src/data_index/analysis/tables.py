import data_index.analysis.warehouse as analysis_warehouse
import data_index.protocols
from data_index.iceberg_config import (
    IcebergTableConfig,
    S3TablesCatalogConfig,
)
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata

IMOS_DATA_LIVE_TABLE = IcebergTableConfig(
    catalog_config=S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
    ),
    namespace="inventory",
    table_name="live",
)

STRUCTURED_METADATA_V5 = IcebergTableConfig(
    catalog_config=analysis_warehouse.DATA_INDEX_CATALOG_CONFIG,
    namespace="data_index",
    table_name="structured_metadata_v5",
)

STRUCTURED_METADATA_V6 = IcebergTableConfig(
    catalog_config=analysis_warehouse.DATA_INDEX_CATALOG_CONFIG,
    namespace="data_index",
    table_name="structured_metadata_v6",
)

LOCAL_STRUCTURED_METADATA_TABLE = IcebergTableConfig(
    catalog_config=analysis_warehouse.ANALYSIS_LOCAL_CATALOG,
    namespace="data_index",
    table_name=f"structured_metadata_v{StructuredMetadata.SCHEMA_VERSION}",
)

LOCAL_UNSTRUCTURED_METADATA_TABLE = IcebergTableConfig(
    catalog_config=analysis_warehouse.ANALYSIS_LOCAL_CATALOG,
    namespace="data_index",
    table_name=f"unstructured_metadata_v{UnstructuredMetadata.SCHEMA_VERSION}",
)

LOCAL_DEAD_LETTER_TABLE = IcebergTableConfig(
    catalog_config=analysis_warehouse.ANALYSIS_LOCAL_CATALOG,
    namespace="data_index",
    table_name=f"dead_letter_v{data_index.protocols.DeadLetter.SCHEMA_VERSION}",
)
