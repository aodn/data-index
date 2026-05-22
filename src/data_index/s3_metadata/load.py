import polars
import pathlib
import prefect


@prefect.task(task_run_name="load-collection-{collection}")
def _load_collection(
    live_inventory_lf,
    collection: str,
    path: pathlib.Path = pathlib.Path(".extract/s3_metadata"),
) -> None:

    path = path / collection
    path.mkdir(parents=True, exist_ok=True)
    live_inventory_lf.filter(polars.col("collection").eq(collection)).sink_parquet(
        path=path / "0.parquet"
    )


@prefect.task
def load(
    live_inventory_lf: polars.LazyFrame,
    path: pathlib.Path = pathlib.Path(".extract/s3_metadata"),
) -> polars.LazyFrame:

    collections = (
        live_inventory_lf.select("collection")
        .unique()
        .collect()
        .get_column("collection")  # Safely extracts the column as a Series
    )

    for collection in collections:
        _load_collection(
            live_inventory_lf,
            collection,
            path=path,
        )
