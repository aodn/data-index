import pathlib

from data_index.iceberg_config import S3TablesCatalogConfig, SqliteCatalogConfig

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
ANALYSIS_LOCAL_WAREHOUSE = REPOSITORY_ROOT / ".load/orchestrate-analysis"
ANALYSIS_LOCAL_CATALOG = SqliteCatalogConfig(
    uri=f"sqlite:///{(ANALYSIS_LOCAL_WAREHOUSE / 'catalog.db').resolve()}",
    warehouse=str(ANALYSIS_LOCAL_WAREHOUSE.resolve()),
)

DATA_INDEX_CATALOG_CONFIG = S3TablesCatalogConfig(
    region="ap-southeast-2",
    arn="arn:aws:s3tables:ap-southeast-2:704910415367:bucket/data-index",
)
