import pathlib

from data_index.s3_metadata import (
    extract,
    transform,
    TableScanConfig,
)

import prefect
import polars

@prefect.task
def get_batch(
    collection: str,
    limit: int | None = None,
) -> polars.DataFrame:
    
    # Pre-Process the S3 Metadata table
    df = (
        polars.scan_parquet(".extract/s3_metadata")
        .filter(
            polars.col("collection").eq(collection)
        )
        .select(
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/"
            ).alias("s3_uri"),
            polars.col("size"),
        )
        .collect()
    )

    # Apply limit as sample
    if limit:
        df = df.sample(n=limit if limit <= len(df) else len(df))
    
    return df

@prefect.flow
def get_batch_from_s3_metadata(
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

    # # Collection Pattern
    # # Second component of the key
    # collection_pattern = r"^[^/]+/(?P<collection>[^/]+)"

    return (
        polars.scan_parquet(live_inventory_parquet_path)
        .filter(
            polars.col("key").str.ends_with(".nc"),
        )
        .select(
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/"
            ).alias("s3_uri"),
            polars.col("size"),
        )
        .sort(
            polars.col("s3_uri"),
        )
        .collect()
    )