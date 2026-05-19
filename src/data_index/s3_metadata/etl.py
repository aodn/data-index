from .extract import extract, TableScanConfig
from .transform import transform
from .load import load

import prefect

@prefect.flow
def etl(
    table_scan_config: TableScanConfig,
):
    inventory_lf = extract(table_scan_config=table_scan_config)
    live_inventory_lf = transform(inventory_lf)
    load(live_inventory_lf)

if __name__ == "__main__":
    etl(
        table_scan_config=TableScanConfig(
            row_filter="key LIKE 'IMOS/%'",
        )
    )