import pathlib
import polars
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

@prefect.flow(
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=32),
)
def pipeline() -> None:

    prefix = "IMOS/AATAMS"
    batch_df = get_batch(
        prefix=prefix,
        limit=1_000,
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
                path=pathlib.Path(".load") / pathlib.path(f"prefix={prefix}") / "structured.parquet"
            ),
            unstructured_sink=UnstructuredParquetSink(
                path=pathlib.Path(".load") / pathlib.path(f"prefix={prefix}") / "unstructured.parquet"
            ),
        )
