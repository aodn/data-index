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
    source_root = tmp_path / "local-dev-volume"
    source_root.mkdir(parents=True, exist_ok=True)

    a = source_root / "a.nc"
    b = source_root / "b.nc"
    c = source_root / "nested" / "c.nc"
    c.parent.mkdir(parents=True, exist_ok=True)

    a.write_bytes(b"a")
    b.write_bytes(b"bb")
    c.write_bytes(b"ccc")
    (source_root / "dup-a.nc").symlink_to(a)
    (source_root / "folder.nc").mkdir()

    source = LocalGlobInventorySource(
        root_path=source_root,
        glob_pattern="**/*.nc",
        bucket="local-dev-volume",
        max_files=2,
    )

    inventory = source.inventory()

    assert inventory.height == 2
    expected_bucket = str(source_root.resolve())
    assert inventory["bucket"].to_list() == [expected_bucket, expected_bucket]
    assert inventory["version_id"].to_list() == ["__LOCAL__", "__LOCAL__"]

    expected_keys = ["a.nc", "b.nc"]
    assert inventory["key"].to_list() == expected_keys
    assert inventory["size"].to_list() == [a.stat().st_size, b.stat().st_size]


def test_derives_bucket_segment_relative_keys(tmp_path: pathlib.Path):
    source_root = tmp_path / "Volumes" / "4tb-0"
    file_path = (
        source_root
        / "imos-data"
        / "IMOS"
        / "Argo"
        / "nmdis"
        / "2901615"
        / "2901615_prof.nc"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"argo")

    source = LocalGlobInventorySource(
        root_path=source_root,
        glob_pattern="**/*_prof.nc",
        bucket="imos-data/",
    )

    inventory = source.inventory()

    assert inventory.height == 1
    expected_bucket = str((source_root / "imos-data").resolve())
    assert inventory["bucket"].to_list() == [expected_bucket]
    assert inventory["key"].to_list() == ["IMOS/Argo/nmdis/2901615/2901615_prof.nc"]
    assert inventory["version_id"].to_list() == ["__LOCAL__"]
    assert inventory["size"].to_list() == [4]


def test_key_uses_first_bucket_segment_when_bucket_repeats(tmp_path: pathlib.Path):
    source_root = tmp_path / "imos-data" / "mirror"
    file_path = (
        source_root
        / "imos-data"
        / "IMOS"
        / "Argo"
        / "nmdis"
        / "2901615"
        / "2901615_prof.nc"
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"argo")

    source = LocalGlobInventorySource(
        root_path=source_root,
        glob_pattern="**/*_prof.nc",
        bucket="imos-data",
    )

    inventory = source.inventory()

    assert inventory.height == 1
    expected_bucket = str((tmp_path / "imos-data").resolve())
    assert inventory["bucket"].to_list() == [expected_bucket]
    assert inventory["key"].to_list() == [
        "mirror/imos-data/IMOS/Argo/nmdis/2901615/2901615_prof.nc"
    ]


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


def test_fails_fast_when_bucket_anchor_missing(tmp_path: pathlib.Path):
    file_path = tmp_path / "sample.nc"
    file_path.write_bytes(b"sample")

    source = LocalGlobInventorySource(
        root_path=tmp_path,
        glob_pattern="**/*.nc",
        bucket="imos-data",
    )

    with pytest.raises(ValueError, match="Bucket anchor segment not found"):
        source.inventory()


def test_rejects_zero_max_files(tmp_path: pathlib.Path):
    with pytest.raises(ValueError):
        LocalGlobInventorySource(
            root_path=tmp_path,
            glob_pattern="**/*.nc",
            bucket="local-dev-volume",
            max_files=0,
        )
