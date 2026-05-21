from data_index.file_fetcher.s3_fetcher import S3Fetcher


def test_returns_one_handle_per_uri():
    fetcher = S3Fetcher()
    uris = ["s3://bucket/a.nc", "s3://bucket/b.nc", "s3://bucket/c.nc"]

    handles = fetcher.fetch(uris)

    assert len(handles) == 3


def test_each_handle_s3_uri_matches_input():
    fetcher = S3Fetcher()
    uris = ["s3://bucket/a.nc", "s3://bucket/b.nc"]

    handles = fetcher.fetch(uris)

    assert {h.s3_uri for h in handles} == set(uris)


def test_returns_empty_list_for_empty_input():
    fetcher = S3Fetcher()

    handles = fetcher.fetch([])

    assert handles == []
