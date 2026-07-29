import typing

import pydantic

from data_index.batch_partitioner import GreedyBatchPartitioner
from data_index.file_fetcher import (
    ConcurrentObstoreFetcher,
    FSSpecFetcher,
    LocalFetcher,
    ObstoreFetcher,
)
from data_index.inventory_source import (
    DeltaIcebergTableInventorySource,
    IcebergTableInventorySource,
    LocalGlobInventorySource,
)
from data_index.metadata_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.sink import (
    DummySink,
    IcebergTableSink,
)

# --- Type Routing ---
type InventorySourceType = type[
    DeltaIcebergTableInventorySource
    | IcebergTableInventorySource
    | LocalGlobInventorySource
]
type BatchPartitionerType = type[GreedyBatchPartitioner]
type FileFetcherType = type[
    FSSpecFetcher | ObstoreFetcher | ConcurrentObstoreFetcher | LocalFetcher
]
type MetadataExtractorType = type[AttributeNetCDFExtractor]
type MetadataSinkType = type[IcebergTableSink | DummySink]

# Runtime instance aliases used by Prefect flow parameter validation.
type InventorySource = typing.Annotated[
    DeltaIcebergTableInventorySource
    | IcebergTableInventorySource
    | LocalGlobInventorySource,
    pydantic.Field(discriminator="type"),
]
type BatchPartitioner = GreedyBatchPartitioner
type FileFetcher = typing.Annotated[
    FSSpecFetcher | ObstoreFetcher | ConcurrentObstoreFetcher | LocalFetcher,
    pydantic.Field(discriminator="type"),
]
type MetadataExtractor = AttributeNetCDFExtractor
type MetadataSink = typing.Annotated[
    IcebergTableSink | DummySink,
    pydantic.Field(discriminator="type"),
]
