import polars
import prefect


@prefect.task
def transform_live_table(
    inventory_lf: polars.LazyFrame,
) -> polars.LazyFrame:
    return (
        # Source
        inventory_lf
        # Filter to netcdf files
        .filter(
            polars.col("key").str.ends_with(".nc"),
        )
        # Filter to the latest version of each object
        .filter(
            polars.col("sequence_number").eq(
                polars.col("sequence_number").max().over(polars.col("key"))
            )
        )
        # Remove deleted objects
        .filter(polars.col("is_delete_marker").eq(False))
        # Add collection partition
        .with_columns(
            polars.col("key")
            .str.extract_groups(r"^[^/]+/(?P<collection>[^/]+)")
            .alias("collection"),
        )
        .unnest(polars.col("collection"))
    )


@prefect.task
def describe_live_table(live_inventory_lf: polars.LazyFrame):
    logger = prefect.get_run_logger()
    logger.info("generating describe...")
    logger.info(live_inventory_lf.select("key", "size").describe())


@prefect.task
def transform(
    inventory_lf: polars.LazyFrame,
) -> polars.LazyFrame:
    inventory_live_lf = transform_live_table(
        inventory_lf,
    )
    describe_live_table(inventory_live_lf)
    return inventory_live_lf
