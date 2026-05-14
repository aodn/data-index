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
from data_index.structured_sink.parquet_sink import ParquetSink

@prefect.flow(
    task_runner=prefect.task_runners.ThreadPoolTaskRunner(max_workers=32),
)
def pipeline() -> None:

    batch_df = (
        polars.scan_parquet(source=pathlib.Path("/Users/thommodin/dev/python-spike-testing/s3-metadata/live-imos-data.inventory.parquet"))
        .select(
            polars.col("bucket"),
            polars.col("key"),
            polars.col("version_id"),
            polars.col("size"),
        )
        .filter(
            polars.col("key").str.starts_with("IMOS"),
            polars.col("key").str.ends_with(".nc"),
        )
        .collect()
        .sample(1_000)
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
            structured_sink=ParquetSink(),
        )
