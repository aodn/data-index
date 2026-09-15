import typing

import obstore.store
import polars
import xarray
import zarr

DATASET = typing.Literal[
    # Hourly lucinda jetty station datasets largely abandoned:
    # https://universitytasmania.sharepoint.com/:x:/r/sites/tier2-imos-AODN-Team/Shared%20Documents/AODN-All/Projects/Year-2025/IMOSDataset2CloudOptimised/IMOSPortalcollections.xlsx?d=w913ca11ff25e41e6b055a00b1da948f9&csf=1&web=1&e=5Qb3Ba
    "station_lucinda_jetty_hourly_wetlabs_wqm",
    "station_lucinda_jetty_hourly_wetlabs_bb9",
    "station_lucinda_jetty_hourly_satlantic_hyperocr",
    "station_lucinda_jetty_dalec_derived_product",
    "station_lucinda_jetty_dalec",
    "station_lucinda_jetty_daily_wetlabs_bb9",
]

DATASET_FILTER: dict[DATASET, polars.Expr] = {
    "station_lucinda_jetty_hourly_wetlabs_wqm": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/WQM-hourly")
        & polars.col("key").str.ends_with(".nc")
    ),
    "station_lucinda_jetty_hourly_wetlabs_bb9": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/BB9-hourly")
        & polars.col("key").str.ends_with(".nc")
    ),
    "station_lucinda_jetty_hourly_satlantic_hyperocr": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/HyperOCR-hourly")
        & polars.col("key").str.contains(r".*FV01.*\.nc$")
    ),
    "station_lucinda_jetty_dalec_derived_product": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/DALEC")
        & polars.col("key").str.contains(r".*FV02.*\.nc$")
    ),
    "station_lucinda_jetty_dalec": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/DALEC")
        & polars.col("key").str.contains(r".*FV01.*\.nc$")
    ),
    "station_lucinda_jetty_daily_wetlabs_bb9": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/BB9-daily")
        & polars.col("key").str.contains(r".*\.nc$")
    ),
}


def get_dataset_objects_df(
    df: polars.DataFrame,
    dataset: DATASET,
) -> polars.DataFrame:
    return (
        df.filter(DATASET_FILTER[dataset])
        .with_columns(
            # Add s3 uri
            polars.concat_str(
                polars.lit("s3:/"),
                polars.col("bucket"),
                polars.col("key"),
                separator="/",
            ).alias("s3_uri"),
            # Split out filename
            polars.col("key").str.split("/").list.last().alias("filename"),
        )
        # Extract start and end date
        .with_columns(
            polars.col("filename")
            .str.extract_groups(
                r"(?P<start_date>\d{8}T\d{6}Z).*?_END-(?P<end_date>\d{8}T\d{6}Z)\.nc$"
            )
            .alias("dates")
        )
        .unnest("dates")
        # Convert start and end date to date type
        .with_columns(
            polars.col("start_date").str.to_date("%Y%m%dT%H%M%SZ"),
            polars.col("end_date").str.to_date("%Y%m%dT%H%M%SZ"),
        )
    )


def get_dataset_xarray_dataset(
    dataset: DATASET,
) -> xarray.Dataset:

    # Initialize the obstore backend for your cloud storage (e.g., AWS S3)
    s3_store = obstore.store.S3Store(
        "aodn-cloud-optimised",
        prefix=f"{dataset}.zarr",
        skip_signature=True,
        region="ap-southeast-2",
    )

    # Wrap it with Zarr's ObjectStore adapter
    store = zarr.storage.ObjectStore(store=s3_store, read_only=True)

    return xarray.open_zarr(store)
