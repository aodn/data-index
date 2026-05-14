import polars
import prefect

from data_index.extract import extract
from data_index.transform import transform
from data_index.load import load
from data_index.file_fetcher import S5CMDFetcher
from data_index.metadata_extractor.netcdf_extractor import NetCDFExtractor
from data_index.structured_sink.parquet_sink import ParquetSink

@prefect.flow
def pipeline() -> None:
    batch_df = polars.DataFrame(data=[
        {
            "s3_uri": "s3://imos-data/IMOS/SRS/SST/ghrsst/L3S-1d/day/2025/20250101032000-ABOM-L3S_GHRSST-SSTskin-AVHRR_D-1d_day.nc",
            "size": 19456,
        },
        {
            "s3_uri": "s3://imos-data/IMOS/Argo/dac/nmdis/2901615/2901615_prof.nc",
            "size": 472064,
        },
    ])

    manifest = extract(
        batch_df=batch_df,
        fetcher=S5CMDFetcher(),
    )
    extraction_results = transform(
        manifest=manifest,
        extractor=NetCDFExtractor(),
    )
    load(
        extraction_results=extraction_results,
        structured_sink=ParquetSink(),
    )
