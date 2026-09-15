import pathlib

from data_index.iceberg_config import SqliteCatalogConfig

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]
ANALYSIS_LOCAL_WAREHOUSE = REPOSITORY_ROOT / ".load/orchestrate-analysis"
ANALYSIS_LOCAL_CATALOG = SqliteCatalogConfig(
    uri=f"sqlite:///{(ANALYSIS_LOCAL_WAREHOUSE / 'catalog.db').resolve()}",
    warehouse=str(ANALYSIS_LOCAL_WAREHOUSE.resolve()),
)
