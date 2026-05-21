import fsspec
import xarray
import cloudpathlib
import rich

import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("fsspec").setLevel(logging.DEBUG)

s3_uris = [
    "s3://imos-data/IMOS/SRS/SST/ghrsst/L3SM-1m/day/2026/20260131032000-ABOM-L3S_GHRSST-SSTskin-MultiSensor-1m_day.nc",
    "s3://imos-data/IMOS/SRS/SST/ghrsst/L3SM-1m/day/2026/20260131032000-ABOM-L3S_GHRSST-SSTskin-MultiSensor-1m_day.nc",
    # "/Users/thommodin/dev/temp/dataflow-orchestration/tests/resources_for_tests/good_with_correct_file_name_structure.nc",
]

class BlockCachedOpenFile(fsspec.core.OpenFile):
    """OpenFile that forwards open_kwargs to fs.open(), enabling BlockCache without mmap overhead."""

    def __init__(self, fs, path, **open_kwargs):
        super().__init__(fs, path)
        self._open_kwargs = open_kwargs

    def __enter__(self):
        f = self.fs.open(self.path, **self._open_kwargs)
        self.fobjects = [f]
        return f


fs = fsspec.filesystem("s3")
open_file = BlockCachedOpenFile(
    fs,
    "imos-data/IMOS/SRS/SST/ghrsst/L3SM-1m/day/2026/20260131032000-ABOM-L3S_GHRSST-SSTskin-MultiSensor-1m_day.nc",
    cache_type="blockcache",
    block_size=1024 * 128,
)

with open_file as f:
    ds = xarray.open_dataset(filename_or_obj=f)

# class FSSpecFetcher:
#     _fs_caching_options = {"cache_type": "blockcache", "block_size": 1024 * 128}

#     def fetch(self, s3_paths: list[cloudpathlib.S3Path]) -> list[fsspec.core.OpenFile]:
#         uris = [path.as_uri() for path in s3_paths]
        
#         # fsspec.open_files creates a list of lazy OpenFile objects
#         return fsspec.open_files(
#             uris, 
#             protocol="s3", 
#             anon=True, 
#             **self._fs_caching_options
#         )



# fsspec.filesystem(
#     protocol="s3",
# )
# rich.print(open_files)

# for open_file in open_files:
#     with open_file as open_file:
#         print(xarray.open_dataset(filename_or_obj=open_file))



# %%time
# uri = "s3://its-live-data/test-space/sample-data/sst.mnmean.nc"

# # If we need to pass credentials to our remote storage we can do it here, in this case this is a public bucket
# fs = fsspec.filesystem('s3', anon=True)

# fsspec_caching = {
#     "cache_type": "blockcache",  # block cache stores blocks of fixed size and uses eviction using a LRU strategy.
#     "block_size": 8
#     * 1024
#     * 1024,  # size in bytes per block, adjust depends on the file size but the recommended size is in the MB
# }

# # we are not using a context, we can use ds until we manually close it.
# ds = xr.open_dataset(fs.open(uri, **fsspec_caching), engine="h5netcdf")
# ds

# of = fsspec.open("blockcache://anaconda-public-datasets/gdelt/csv/20150906.export.csv", 
#                  mode='rt', target_protocol='s3', cache_storage='/tmp/cache2',
#                  target_options={'anon': True, "default_block_size": 2**20})

# fsspec.open(
#     urlpath="s3://imos-data/IMOS/SRS/SST/ghrsst/L3SM-1m/day/2026/20260131032000-ABOM-L3S_GHRSST-SSTskin-MultiSensor-1m_day.nc",
#     **
# )
# # Example setting a 10 MB block size for reading
# ds = xarray.open_dataset(
#     "s3://imos-data/IMOS/SRS/SST/ghrsst/L3SM-1m/day/2026/20260131032000-ABOM-L3S_GHRSST-SSTskin-MultiSensor-1m_day.nc",
#     backend_kwargs={
#         "storage_options": {
#             "s3": {
#                 "cache_type": "block", # Choices: "readahead", "block", "none"
#                 "block_size": 1024 * 128 # Block size in bytes (10 MB in this example)
#             }
#         }
#     }
# )