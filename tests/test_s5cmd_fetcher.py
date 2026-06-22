import inspect
import pathlib
from unittest.mock import patch

import pytest

from data_index.file_fetcher.s5cmd_fetcher import S5CMDFetcher
from data_index.protocols import BatchEntry, ObjectReference


def _entry(uri: str, version_id: str = "v1") -> BatchEntry:
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return BatchEntry(
        object_ref=ObjectReference(bucket=bucket, key=key, version_id=version_id)
    )


# --- Protocol conformance ---


def test_fetch_signature_matches_file_fetcher_protocol():
    """S5CMDFetcher.fetch() must accept exactly (self, entries: list[BatchEntry]) to satisfy FileFetcher."""
    sig = inspect.signature(S5CMDFetcher.fetch)
    params = list(sig.parameters.keys())
    assert params == ["self", "entries"], (
        f"S5CMDFetcher.fetch() has unexpected signature params: {params}. "
        "FileFetcher protocol requires exactly (self, entries)."
    )


# --- Static helpers ---


def test_prepare_commands_generates_cp_commands(tmp_path):
    entries = [_entry("s3://bucket/a.nc", "v1"), _entry("s3://bucket/b.nc", "v2")]
    commands, destination_object_refs = S5CMDFetcher._prepare_commands(
        entries, extract_path=tmp_path
    )
    token_v1 = S5CMDFetcher._version_path_token("v1")
    token_v2 = S5CMDFetcher._version_path_token("v2")

    assert len(commands) == 2
    assert commands[0].startswith('cp --version-id "v1" s3://bucket/a.nc')
    assert commands[1].startswith('cp --version-id "v2" s3://bucket/b.nc')
    assert (
        str((tmp_path / "bucket" / token_v1 / "a.nc").resolve())
        in destination_object_refs
    )
    assert (
        str((tmp_path / "bucket" / token_v2 / "b.nc").resolve())
        in destination_object_refs
    )


def test_prepare_commands_uses_filesystem_safe_version_token(tmp_path):
    version_id = 'v:1/unsafe"chars\\here'
    entries = [_entry("s3://bucket/a.nc", version_id)]
    commands, destination_object_refs = S5CMDFetcher._prepare_commands(
        entries, extract_path=tmp_path
    )
    token = S5CMDFetcher._version_path_token(version_id)

    assert token in commands[0]
    assert version_id not in str(next(iter(destination_object_refs.keys())))
    assert '--version-id "v:1/unsafe\\"chars\\\\here"' in commands[0]


def test_parse_s5cmd_output_returns_handles_for_cp_lines(tmp_path):
    """Each 'cp s3://... local/path' line must produce one XarrayHandle."""
    token_v1 = S5CMDFetcher._version_path_token("v1")
    token_v2 = S5CMDFetcher._version_path_token("v2")
    local_a = (tmp_path / "bucket" / token_v1 / "a.nc").resolve()
    local_b = (tmp_path / "bucket" / token_v2 / "b.nc").resolve()
    stdout = f"cp s3://bucket/a.nc {local_a}\ncp s3://bucket/b.nc {local_b}\n"
    destination_object_refs = {
        str(local_a): ObjectReference(bucket="bucket", key="a.nc", version_id="v1"),
        str(local_b): ObjectReference(bucket="bucket", key="b.nc", version_id="v2"),
    }

    handles = S5CMDFetcher._parse_s5cmd_output(stdout, destination_object_refs)

    assert len(handles) == 2
    refs = {h.object_ref for h in handles}
    assert ObjectReference(bucket="bucket", key="a.nc", version_id="v1") in refs
    assert ObjectReference(bucket="bucket", key="b.nc", version_id="v2") in refs


def test_parse_s5cmd_output_ignores_non_cp_lines():
    stdout = "INFO starting\ncp s3://bucket/a.nc /tmp/a.nc\nINFO done\n"
    destination_object_refs = {
        str(pathlib.Path("/tmp/a.nc").resolve()): ObjectReference(
            bucket="bucket", key="a.nc", version_id="v1"
        )
    }

    handles = S5CMDFetcher._parse_s5cmd_output(stdout, destination_object_refs)

    assert len(handles) == 1


# --- fetch() behaviour with mocked s5cmd ---


@pytest.fixture
def fetcher():
    with patch("data_index.file_fetcher.s5cmd_fetcher.sh") as mock_sh:
        mock_sh.s5cmd.return_value = "s5cmd version 2.0.0"
        yield S5CMDFetcher(), mock_sh


def test_fetch_returns_empty_list_for_empty_uris(fetcher):
    instance, mock_sh = fetcher
    handles = instance.fetch([])

    assert handles == []
    mock_sh.s5cmd.assert_called_once_with("version")  # only the availability check


def test_fetch_calls_s5cmd_run_with_cp_commands(tmp_path, fetcher):
    instance, mock_sh = fetcher
    instance.extract_path = tmp_path
    token_v1 = S5CMDFetcher._version_path_token("v1")
    local_a = (tmp_path / "bucket" / token_v1 / "a.nc").resolve()
    mock_sh.s5cmd.return_value = f"cp s3://bucket/a.nc {local_a}"

    instance.fetch([_entry("s3://bucket/a.nc")])

    run_calls = [c for c in mock_sh.s5cmd.call_args_list if c.args and "run" in c.args]
    assert len(run_calls) == 1
