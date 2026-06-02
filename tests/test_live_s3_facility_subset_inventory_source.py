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
            {"bucket": "imos-data", "key": "IMOS/ACORN/a1.nc", "size": 1},
            {"bucket": "imos-data", "key": "IMOS/ACORN/a2.nc", "size": 2},
            {"bucket": "imos-data", "key": "IMOS/ACORN/a3.nc", "size": 3},
            {"bucket": "imos-data", "key": "IMOS/ANMN/b1.nc", "size": 4},
            {"bucket": "imos-data", "key": "IMOS/ANMN/b2.nc", "size": 5},
            {"bucket": "imos-data", "key": "IMOS/ANMN/b3.nc", "size": 6},
            {"bucket": "imos-data", "key": "misc/other.nc", "size": 7},
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

    assert set(df.columns) == {"s3_uri", "size"}
    assert len(df.filter(polars.col("s3_uri").str.contains("/IMOS/ACORN/"))) == 2
    assert len(df.filter(polars.col("s3_uri").str.contains("/IMOS/ANMN/"))) == 2
    assert not df["s3_uri"].str.contains("/misc/").any()


def test_inventory_subset_excludes_non_facility_paths(tmp_path: pathlib.Path):
    path = tmp_path / "s3_metadata"
    _write_inventory(
        path,
        [
            {"bucket": "imos-data", "key": "IMOS/ACORN/a1.nc", "size": 1},
            {"bucket": "imos-data", "key": "misc/other.nc", "size": 2},
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
    assert df["s3_uri"].to_list() == ["s3://imos-data/IMOS/ACORN/a1.nc"]
