import pathlib

import polars as pl
import pytest

from data_index.inventory_source.local_glob import LocalGlobInventorySource


def test_raises_for_missing_root_path(tmp_path: pathlib.Path):
    with pytest.raises(ValueError, match="root_path does not exist"):
        LocalGlobInventorySource(
            root_path=tmp_path / "missing",
            glob_pattern="**/*.nc",
            bucket="local-dev-volume",
        )


def test_returns_empty_typed_inventory_when_no_matches(tmp_path: pathlib.Path):
    source = LocalGlobInventorySource(
        root_path=tmp_path,
        glob_pattern="**/*.nc",
        bucket="local-dev-volume",
    )

    inventory = source.inventory()

    assert inventory.is_empty()
    assert inventory.columns == ["bucket", "key", "version_id", "size"]
    assert inventory.dtypes == [pl.String, pl.String, pl.String, pl.Int64]


def test_inventory_is_deduped_sorted_and_capped(tmp_path: pathlib.Path):
    a = tmp_path / "a.nc"
    b = tmp_path / "b.nc"
    c = tmp_path / "nested" / "c.nc"
    c.parent.mkdir(parents=True, exist_ok=True)

    a.write_bytes(b"a")
    b.write_bytes(b"bb")
    c.write_bytes(b"ccc")
    (tmp_path / "dup-a.nc").symlink_to(a)
    (tmp_path / "folder.nc").mkdir()

    source = LocalGlobInventorySource(
        root_path=tmp_path,
        glob_pattern="**/*.nc",
        bucket="local-dev-volume",
        max_files=2,
    )

    inventory = source.inventory()

    assert inventory.height == 2
    assert inventory["bucket"].to_list() == ["local-dev-volume", "local-dev-volume"]
    assert inventory["version_id"].to_list() == ["__LOCAL__", "__LOCAL__"]

    expected_keys = [str(a.resolve()), str(b.resolve())]
    assert inventory["key"].to_list() == expected_keys
    assert inventory["size"].to_list() == [a.stat().st_size, b.stat().st_size]


def test_rejects_symlink_target_outside_root(tmp_path: pathlib.Path):
    outside = tmp_path.parent / "outside.nc"
    outside.write_bytes(b"outside")

    (tmp_path / "escape.nc").symlink_to(outside)

    source = LocalGlobInventorySource(
        root_path=tmp_path,
        glob_pattern="**/*.nc",
        bucket="local-dev-volume",
    )

    with pytest.raises(ValueError, match="escapes root_path"):
        source.inventory()


def test_rejects_zero_max_files(tmp_path: pathlib.Path):
    with pytest.raises(ValueError):
        LocalGlobInventorySource(
            root_path=tmp_path,
            glob_pattern="**/*.nc",
            bucket="local-dev-volume",
            max_files=0,
        )
