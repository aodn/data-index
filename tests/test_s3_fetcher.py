from data_index.file_fetcher.s3_fetcher import S3Fetcher
from data_index.protocols import BatchEntry, ObjectReference


def _entry(uri: str, version_id: str = "v1") -> BatchEntry:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return BatchEntry(
        object_ref=ObjectReference(bucket=bucket, key=key, version_id=version_id)
    )


def test_returns_one_handle_per_uri():
    fetcher = S3Fetcher()
    entries = [
        _entry(uri=u)
        for u in ["s3://bucket/a.nc", "s3://bucket/b.nc", "s3://bucket/c.nc"]
    ]

    handles = fetcher.fetch(entries)

    assert len(handles) == 3


def test_each_handle_s3_uri_matches_input():
    fetcher = S3Fetcher()
    uris = ["s3://bucket/a.nc", "s3://bucket/b.nc"]
    entries = [_entry(uri=u) for u in uris]

    handles = fetcher.fetch(entries)

    assert {h.s3_uri for h in handles} == set(uris)


def test_returns_empty_list_for_empty_input():
    fetcher = S3Fetcher()

    handles = fetcher.fetch([])

    assert handles == []
