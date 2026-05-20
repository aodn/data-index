import pathlib

import boto3
import cloudpathlib

from data_index.protocols import ManifestEntry


class S3Fetcher:
    """FileFetcher implementation that downloads files from S3 using boto3."""

    def fetch(self, uris: list[str], extract_path: pathlib.Path) -> list[ManifestEntry]:
        s3 = boto3.client("s3")
        entries: list[ManifestEntry] = []
        for uri in uris:
            s3_path = cloudpathlib.S3Path(uri)
            local_path = extract_path / s3_path.bucket / s3_path.key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(s3_path.bucket, s3_path.key, str(local_path))
            entries.append(ManifestEntry(s3_uri=uri, target=local_path.resolve()))
        return entries
