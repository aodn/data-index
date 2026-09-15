from data_index.iceberg_config import IcebergTableConfig, S3TablesCatalogConfig

IMOS_DATA_LIVE_TABLE = IcebergTableConfig(
    catalog_config=S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/imos-data-inventory",
    ),
    namespace="inventory",
    table_name="live",
)

STRUCTURED_METADATA_V5 = IcebergTableConfig(
    catalog_config=S3TablesCatalogConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
    ),
    namespace="data_index",
    table_name="structured_metadata_v5",
)
