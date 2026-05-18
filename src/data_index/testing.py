import pathlib

from data_index.s3_metadata import (
    extract,
    transform,
    TableScanConfig,
)

import prefect
import polars

@prefect.flow
def get_batch(
    prefix: str,
    limit: int | None = 10_000,
) -> polars.DataFrame:
    
    inventory_parquet_path = pathlib.Path(".extract") / pathlib.Path(prefix) / pathlib.Path(f"limit={limit}/imos-data.inventory.parquet")
    live_inventory_parquet_path = pathlib.Path(".extract") / pathlib.Path(prefix) / pathlib.Path(f"limit={limit}/live-imos-data.inventory.parquet")
    extract(
        table_scan_config=TableScanConfig(
            limit=limit,
            row_filter=f"key LIKE '{prefix}%'",
        ),
        inventory_parquet_path=inventory_parquet_path,
    )

    transform(
        inventory_parquet_path=inventory_parquet_path,
        live_inventory_parquet_path=live_inventory_parquet_path,
    )

    # Collection Pattern
    # Second component of the key
    collection_pattern = r"^[^/]+/(?P<collection>[^/]+)"

    return (
        polars.scan_parquet(live_inventory_parquet_path)
        .filter(
            polars.col("key").str.ends_with(".nc"),
        )
        .select(
            polars.col("key").str.extract_groups(collection_pattern).alias("collection"),
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/"
            ).alias("s3_uri"),
            polars.col("size"),
        )
        .unnest(polars.col("collection"))
        .sort(
            polars.col("s3_uri"),
        )
        .collect()
    )

if __name__ == "__main__":

    batch = get_batch(
        prefix="IMOS/AATAMS/",
        limit=1,
    )

    print(batch)