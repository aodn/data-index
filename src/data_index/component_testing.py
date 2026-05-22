import pathlib
import prefect
import prefect.task_runners

from data_index.extract import extract
from data_index.transform import transform
from data_index.load import load
from data_index.file_fetcher import S3Fetcher, S5CMDFetcher, ThresholdFileFetcher
from data_index.metadata_extractor import NetCDFExtractor, UnstructuedNetCDFExtractor
from data_index.structured_sink import StructuredParquetSink
from data_index.unstructured_sink import UnstructuredParquetSink
from data_index.unstructured_metadata import InMemoryUnstructuredMetadata
from data_index.testing import get_batch, get_threshold_batch, get_batch_filtered
import polars

import logging
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("fsspec").setLevel(logging.DEBUG)

IMOS_DATA_BUCKET_PREFIXES = [
    # Processed
    # "IMOS/AATAMS",
    # "IMOS/ACORN_JCU_historical",
    # "IMOS/ACORN",
    # "IMOS/ANFOG",
    # "IMOS/Argo",
    # "IMOS/AUV",
    # "IMOS/BGC_DB",
    # "IMOS/COASTAL-WAVE-BUOYS",
    # "IMOS/DWM",
    # "IMOS/eMII",
    # "IMOS/FAIMMS",
    # "IMOS/NRMN",
    # "IMOS/NTP",
    # "IMOS/OceanCurrent",
    # "IMOS/SAIMOS",
    # "IMOS/SOOP",
    "IMOS/SRS",
    # "IMOS/ANMN",
]

@prefect.flow(
    flow_run_name="pipeline-{prefix}",
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=32)
)
def pipeline(
    prefix: str,
    limit: int,
) -> None:

    # batch_df = get_batch(
    #     collection=prefix.split("/")[-1],
    #     limit=limit,
    # )

    # batch_df = get_threshold_batch(threshold=1024 ** 2 * 10)

    batch_df = get_batch_filtered(
        expressions=[polars.col("size").le(1024 ** 2)],
        limit=10_000,
    )

    xarray_handles = extract(
        batch_df=batch_df,
        fetcher=ThresholdFileFetcher(
            size_threshold_bytes=1024 ** 2 * 10, # 10mb
            disk_fetcher=S5CMDFetcher(),
            cloud_fetcher=S3Fetcher(
                block_size=1024 ** 2 * 5, # 5mb
            ),
        ),
    )
    extraction_results = transform(
        xarray_handles=xarray_handles,
        extractor=UnstructuedNetCDFExtractor(),
        metadata_factory=InMemoryUnstructuredMetadata,
    )
    load(
        extraction_results=extraction_results,
        structured_sink=StructuredParquetSink(
            path=pathlib.Path(".load/structured_metadata") / pathlib.Path(f"prefix={prefix.removeprefix('IMOS/')}") / "0.parquet"
        ),
        unstructured_sink=UnstructuredParquetSink(
            path=pathlib.Path(".load/unstructured_metadata") / pathlib.Path(f"prefix={prefix.removeprefix('IMOS/')}") / "0.parquet"
        ),
    )

@prefect.flow
def main(
    prefixes: list[str] = IMOS_DATA_BUCKET_PREFIXES,
    limit: int = 100,
) -> None:
    
    for prefix in prefixes:
        pipeline(
            prefix=prefix,
            limit=limit,
        )
    