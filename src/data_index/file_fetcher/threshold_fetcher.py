import prefect.artifacts

from data_index.protocols import BatchEntry, FileFetcher, XarrayHandle
from data_index.file_fetcher.s3_fetcher import S3Fetcher


class ThresholdFileFetcher:
    """FileFetcher that routes each entry to a disk or cloud fetcher based on file size.

    Entries with size_bytes < size_threshold_bytes are routed to disk_fetcher (full download).
    Entries with size_bytes >= size_threshold_bytes, or with size_bytes=None, are routed to
    cloud_fetcher (byte-range reads via fsspec — no up front download required).
    """

    def __init__(
        self,
        size_threshold_bytes: int,
        disk_fetcher: FileFetcher,
        cloud_fetcher: FileFetcher | None = None,
    ):
        self._threshold = size_threshold_bytes
        self._disk_fetcher = disk_fetcher
        self._cloud_fetcher = cloud_fetcher or S3Fetcher()

    def _is_small(self, entry: BatchEntry) -> bool:
        return entry.size_bytes is not None and entry.size_bytes < self._threshold

    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:
        if not entries:
            return []

        disk_entries = [e for e in entries if self._is_small(e)]
        cloud_entries = [e for e in entries if not self._is_small(e)]

        results: list[XarrayHandle] = []
        if disk_entries:
            results.extend(self._disk_fetcher.fetch(disk_entries))
        if cloud_entries:
            results.extend(self._cloud_fetcher.fetch(cloud_entries))

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
            description=f"Routing decisions (threshold: {self._threshold:,} bytes)",
        )

        return results
