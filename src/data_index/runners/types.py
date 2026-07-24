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
InventorySource: type[DeltaIcebergTableInventorySource | IcebergTableInventorySource]
BatchPartitioner: type[GreedyBatchPartitioner]
FileFetcher: type[FSSpecFetcher | ObstoreFetcher | ConcurrentObstoreFetcher]
MetadataExtractor: type[AttributeNetCDFExtractor]
MetadataSink: type[IcebergTableSink | DummySink]
