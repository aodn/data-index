import typing

from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import (
    ConcurrentObstoreFetcher,
    FSSpecFetcher,
    ObstoreFetcher,
)
from data_index.inventory_source import (
    DeltaIcebergTableInventorySource,
    IcebergTableInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.sink import (
    DummySink,
    IcebergTableSink,
)

# --- Type Routing ---
InventorySource: typing.TypeAlias = (
    DeltaIcebergTableInventorySource | IcebergTableInventorySource
)
BatchPartitioner: typing.TypeAlias = GreedyBatchPartitioner
FileFetcher: typing.TypeAlias = (
    FSSpecFetcher | ObstoreFetcher | ConcurrentObstoreFetcher
)
MetadataExtractor: typing.TypeAlias = AttributeNetCDFExtractor
MetadataSink: typing.TypeAlias = IcebergTableSink | DummySink
