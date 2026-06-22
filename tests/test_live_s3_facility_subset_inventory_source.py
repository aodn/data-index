import pathlib

import polars

from data_index.inventory_source.live_s3_facility_subset import (
    LiveS3InventorySourceFacilitySubset,
)


def _write_inventory(path: pathlib.Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    polars.DataFrame(rows).write_parquet(path / "part-0.parquet")


def test_inventory_subsets_per_facility(tmp_path: pathlib.Path):
    path = tmp_path / "s3_metadata"
    _write_inventory(
        path,
        [
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a1.nc",
                "version_id": "v1",
                "size": 1,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a2.nc",
                "version_id": "v2",
                "size": 2,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a3.nc",
                "version_id": "v3",
                "size": 3,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b1.nc",
                "version_id": "v4",
                "size": 4,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b2.nc",
                "version_id": "v5",
                "size": 5,
            },
            {
                "bucket": "imos-data",
                "key": "IMOS/ANMN/b3.nc",
                "version_id": "v6",
                "size": 6,
            },
            {
                "bucket": "imos-data",
                "key": "misc/other.nc",
                "version_id": "v7",
                "size": 7,
            },
        ],
    )

    source = LiveS3InventorySourceFacilitySubset.model_construct(
        table_config=None,
        table_scan_config=None,
        path=path,
        skip_if_exists=True,
        subset_per_facility=2,
    )

    df = source.inventory()

    assert set(df.columns) == {"bucket", "key", "version_id", "size"}
    assert len(df.filter(polars.col("key").str.contains("^IMOS/ACORN/"))) == 2
    assert len(df.filter(polars.col("key").str.contains("^IMOS/ANMN/"))) == 2
    assert not df["key"].str.contains("^misc/").any()


def test_inventory_subset_excludes_non_facility_paths(tmp_path: pathlib.Path):
    path = tmp_path / "s3_metadata"
    _write_inventory(
        path,
        [
            {
                "bucket": "imos-data",
                "key": "IMOS/ACORN/a1.nc",
                "version_id": "v1",
                "size": 1,
            },
            {
                "bucket": "imos-data",
                "key": "misc/other.nc",
                "version_id": "v2",
                "size": 2,
            },
        ],
    )

    source = LiveS3InventorySourceFacilitySubset.model_construct(
        table_config=None,
        table_scan_config=None,
        path=path,
        skip_if_exists=True,
        subset_per_facility=1,
    )

    df = source.inventory()

    assert len(df) == 1
    assert df["key"].to_list() == ["IMOS/ACORN/a1.nc"]
