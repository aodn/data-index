import prefect

from .extract import IcebergTableConfig, TableScanConfig, extract
from .load import load
from .transform import transform


@prefect.flow
def etl(
    table_config: IcebergTableConfig,
    table_scan_config: TableScanConfig = TableScanConfig(),
):
    inventory_lf = extract(
        table_config=table_config, table_scan_config=table_scan_config
    )
    live_inventory_lf = transform(inventory_lf)
    load(live_inventory_lf)


if __name__ == "__main__":
    from data_index.iceberg_config import S3TablesCatalogConfig

    etl(
        table_config=IcebergTableConfig(
            catalog_config=S3TablesCatalogConfig(
                region="ap-southeast-2",
                arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
            ),
            namespace="b_imos-data",
            table_name="inventory",
        ),
        table_scan_config=TableScanConfig(
            row_filter="key LIKE 'IMOS/%'",
        ),
    )
