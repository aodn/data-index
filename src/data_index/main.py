import pathlib
import prefect
import prefect.task_runners
import tempfile

from data_index.extract import extract
from data_index.transform import transform
from data_index.load import load
from data_index.file_fetcher import S5CMDFetcher
from data_index.metadata_extractor.netcdf_extractor import NetCDFExtractor
from data_index.structured_sink import StructuredParquetSink
from data_index.unstructured_sink import UnstructuredParquetSink
from data_index.testing import get_batch

IMOS_DATA_BUCKET_PREFIXES = [
    # Processed
    # "IMOS/AATAMS",
    # "IMOS/ACORN_JCU_historical",
    # "IMOS/ACORN",
    # "IMOS/ANFOG",
    "IMOS/ANMN",
    "IMOS/Argo",
    "IMOS/AUV",
    "IMOS/BGC_DB",
    "IMOS/COASTAL-WAVE-BUOYS",
    "IMOS/DWM",
    "IMOS/eMII",
    "IMOS/FAIMMS",
    "IMOS/NRMN",
    "IMOS/NTP",
    "IMOS/OceanCurrent",
    "IMOS/SAIMOS",
    "IMOS/SOOP",
    "IMOS/SRS",
]

@prefect.flow(
    flow_run_name="pipeline-{prefix}",
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=32),
)
def pipeline(
    prefix: str,
    limit: int,
) -> None:

    batch_df = get_batch(
        prefix=prefix,
        limit=limit,
    )

    with tempfile.TemporaryDirectory() as temporary_directory:

        manifest = extract(
            batch_df=batch_df,
            fetcher=S5CMDFetcher(),
            extract_path=pathlib.Path(temporary_directory),
        )
        extraction_results = transform(
            manifest=manifest,
            extractor=NetCDFExtractor(),
        )
        load(
            extraction_results=extraction_results,
            structured_sink=StructuredParquetSink(
                path=pathlib.Path(".load/structured_metadata") / pathlib.Path(f"prefix={prefix.lstrip("IMOS/")}") / "0.parquet"
            ),
            unstructured_sink=UnstructuredParquetSink(
                path=pathlib.Path(".load/unstructured_metadata") / pathlib.Path(f"prefix={prefix.lstrip("IMOS/")}") / "0.parquet"
            ),
        )

@prefect.flow
def main(
    prefixes: list[str] = IMOS_DATA_BUCKET_PREFIXES,
    limit: int = 1_000,
) -> None:
    
    for prefix in prefixes:
        pipeline(
            prefix=prefix,
            limit=limit,
        )
    