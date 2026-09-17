import typing

import boto3
import obstore.store
import obstore.auth.boto3
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
    "station_lucinda_jetty_daily_wetlabs_acs",
    "station_lucinda_jetty_daily_satlantic_hyperocr"
    "station_lucinda_jetty_daily_satlantic_hyperocr_derived_product",
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
    "station_lucinda_jetty_daily_wetlabs_acs": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/ACS-daily")
        & polars.col("key").str.contains(r".*\.nc$")
    ),
    "station_lucinda_jetty_daily_satlantic_hyperocr": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/HyperOCR-daily")
        & polars.col("key").str.contains(r".*FV01.*\.nc$")
    ),
    "station_lucinda_jetty_daily_satlantic_hyperocr_derived_product": (
        polars.col("key").str.contains("IMOS/SRS/OC/LJCO/HyperOCR-daily")
        & polars.col("key").str.contains(r".*FV02.*\.nc$")
    ),
}


def get_dataset_objects_df(
    df: polars.DataFrame,
    dataset: DATASET,
    backend: typing.Literal["s3", "disk"] = "s3",
) -> polars.DataFrame:

    uri_expression = [
        polars.col("bucket"),
        polars.col("key"),
    ]

    match backend:
        case "s3":
            uri_expression.insert(0, polars.lit("s3:/"))
        case "disk":
            pass
        case _:
            raise NotImplementedError

    return (
        df.filter(DATASET_FILTER[dataset])
        .with_columns(
            # Add s3 uri
            polars.concat_str(
                exprs=uri_expression,
                separator="/",
            ).alias("uri"),
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
    dataset: str,
    bucket: str = "aodn-cloud-optimised",
    skip_signature: bool = True,
    profile_name: str = "edge-projectofficer",
) -> xarray.Dataset:

    # Set up dynamic required s3 store configuration
    s3_store_config = {
        "prefix": f"{dataset}.zarr",
        "region": "ap-southeast-2",
    }

    # Set up credentials
    if skip_signature:
        s3_store_config["skip_signature"] = True

    else:
        session = boto3.Session(
            profile_name=profile_name,
            region_name="ap-southeast-2",
        )
        credential_provider = obstore.auth.boto3.Boto3CredentialProvider(session)
        s3_store_config["credential_provider"]=credential_provider

    s3_store = obstore.store.S3Store(
        bucket,
        **s3_store_config,
    )

    # Wrap it with Zarr's ObjectStore adapter
    store = zarr.storage.ObjectStore(store=s3_store, read_only=True)

    return xarray.open_zarr(store)