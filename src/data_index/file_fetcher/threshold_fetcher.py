import typing

import prefect.artifacts
import pydantic

from data_index.file_fetcher import S3Fetcher, S5CMDFetcher
from data_index.protocols import BatchEntry, XarrayHandle


class ThresholdFileFetcher(pydantic.BaseModel):
    """FileFetcher that routes each entry to a disk or cloud fetcher based on file size.

    Entries with size_bytes < size_threshold_bytes are routed to disk_fetcher (full download).
    Entries with size_bytes >= size_threshold_bytes, or with size_bytes=None, are routed to
    cloud_fetcher (byte-range reads via fsspec — no up front download required).
    """

    type: typing.Literal["threshold_fetcher"] = pydantic.Field(
        default="threshold_fetcher"
    )

    size_threshold_bytes: int
    disk_fetcher: S5CMDFetcher
    cloud_fetcher: S3Fetcher

    def _is_small(self, entry: BatchEntry) -> bool:
        return (
            entry.size_bytes is not None
            and entry.size_bytes < self.size_threshold_bytes
        )

    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:
        if not entries:
            return []

        disk_entries = [e for e in entries if self._is_small(e)]
        cloud_entries = [e for e in entries if not self._is_small(e)]

        results: list[XarrayHandle] = []
        if disk_entries:
            results.extend(self.disk_fetcher.fetch(disk_entries))
        if cloud_entries:
            results.extend(self.cloud_fetcher.fetch(cloud_entries))

        prefect.artifacts.create_table_artifact(
            key="threshold-fetcher-routing",
            table=[
                {"s3_uri": e.uri, "size_bytes": e.size_bytes, "route": "disk"}
                for e in disk_entries
            ]
            + [
                {"s3_uri": e.uri, "size_bytes": e.size_bytes, "route": "cloud"}
                for e in cloud_entries
            ],
            description=f"Routing decisions (threshold: {self.size_threshold_bytes:,} bytes)",
        )

        return results
