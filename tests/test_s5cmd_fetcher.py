import inspect
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from data_index.file_fetcher.s5cmd_fetcher import S5CMDFetcher
from data_index.protocols import BatchEntry


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
    uris = ["s3://bucket/a.nc", "s3://bucket/b.nc"]
    commands = S5CMDFetcher._prepare_commands(uris, extract_path=tmp_path)

    assert len(commands) == 2
    assert commands[0].startswith("cp s3://bucket/a.nc")
    assert commands[1].startswith("cp s3://bucket/b.nc")


def test_parse_s5cmd_output_returns_handles_for_cp_lines(tmp_path):
    """Each 'cp s3://... local/path' line must produce one XarrayHandle."""
    local_a = tmp_path / "bucket" / "a.nc"
    local_b = tmp_path / "bucket" / "b.nc"
    stdout = f"cp s3://bucket/a.nc {local_a}\ncp s3://bucket/b.nc {local_b}\n"

    handles = S5CMDFetcher._parse_s5cmd_output(stdout)

    assert len(handles) == 2
    uris = {h.s3_uri for h in handles}
    assert "s3://bucket/a.nc" in uris
    assert "s3://bucket/b.nc" in uris


def test_parse_s5cmd_output_ignores_non_cp_lines():
    stdout = "INFO starting\ncp s3://bucket/a.nc /tmp/a.nc\nINFO done\n"

    handles = S5CMDFetcher._parse_s5cmd_output(stdout)

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
    local_a = tmp_path / "bucket" / "a.nc"
    mock_sh.s5cmd.return_value = f"cp s3://bucket/a.nc {local_a}"

    instance.fetch([BatchEntry(uri="s3://bucket/a.nc")])

    run_calls = [c for c in mock_sh.s5cmd.call_args_list if c.args and c.args[0] == "run"]
    assert len(run_calls) == 1
