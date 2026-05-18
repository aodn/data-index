import pathlib

import polars
import prefect


@prefect.task
def sink_transform_live_table(
    inventory_parquet_path: pathlib.Path,
    live_inventory_parquet_path: pathlib.Path,
):
    
    (
        # Source
        polars.scan_parquet(inventory_parquet_path)

        # Filter to the latest version of each object
        .filter(
            polars.col("sequence_number").eq(
                polars.col("sequence_number").max().over(polars.col("key"))
            )
        )

        # Remove deleted objects
        .filter(polars.col("is_delete_marker").eq(False))
        
        # Sink
        .sink_parquet(live_inventory_parquet_path)
    )
    
@prefect.task
def describe_live_table(live_inventory_parquet_path: pathlib.Path):

    logger = prefect.get_run_logger()
    logger.info("generating describe...")
    logger.info(
        polars.scan_parquet(live_inventory_parquet_path)
        .select("key", "size")
        .describe()
    )


@prefect.task
def transform(
    inventory_parquet_path: pathlib.Path = pathlib.Path("imos-data.inventory.parquet"),
    live_inventory_parquet_path: pathlib.Path = pathlib.Path("live-imos-data.inventory.parquet"),
):
    
    sink_transform_live_table(
        inventory_parquet_path,
        live_inventory_parquet_path,
    )

    # describe_live_table(live_inventory_parquet_path)





if __name__ == "__main__":
    transform()