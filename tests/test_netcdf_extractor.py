import json

import numpy
import pytest
import xarray

from data_index.metadata_extractor.netcdf_extractor import NetCDFExtractor
from data_index.protocols import ObjectReference


def make_dataset(
    lat=None,
    lon=None,
    time=None,
    global_attrs=None,
    grid_mapping_var=None,
    grid_mapping_var_attrs=None,
) -> xarray.Dataset:
    coords = {}
    if lat is not None:
        coords["latitude"] = (["latitude"], numpy.array(lat, dtype=float))
    if lon is not None:
        coords["longitude"] = (["longitude"], numpy.array(lon, dtype=float))
    if time is not None:
        coords["time"] = (["time"], numpy.array(time))

    ds = xarray.Dataset(coords=coords, attrs=global_attrs or {})

    if grid_mapping_var is not None:
        ds[grid_mapping_var] = xarray.Variable(
            [], 0, attrs=grid_mapping_var_attrs or {}
        )
        data_var = xarray.Variable(
            ["latitude"],
            numpy.zeros(len(lat or [1])),
            attrs={"grid_mapping": grid_mapping_var},
        )
        ds["data"] = data_var

    return ds


class StubHandle:
    """Minimal XarrayHandle stub for extractor tests."""

    def __init__(
        self,
        ds: xarray.Dataset,
        s3_uri: str = "s3://bucket/file.nc",
        file_format: str | None = None,
    ):
        self._ds = ds
        bucket, key = s3_uri.removeprefix("s3://").split("/", 1)
        self.object_ref = ObjectReference(bucket=bucket, key=key, version_id="v1")
        self.file_format = file_format

    @property
    def ds(self) -> xarray.Dataset:
        return self._ds

    @property
    def s3_uri(self) -> str:
        return self.object_ref.as_uri()

    def cleanup(self) -> None:
        pass


@pytest.fixture
def extractor():
    return NetCDFExtractor()


# --- Structured metadata extraction ---


def test_extracts_lat_lon_time_ranges(extractor):
    ds = make_dataset(
        lat=[-10.0, 0.0, 10.0],
        lon=[100.0, 110.0],
        time=[numpy.datetime64("2020-01-01"), numpy.datetime64("2020-06-01")],
    )
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    assert result.status == "succeeded"
    sm = result.structured_metadata
    assert sm.geospatial_lat_min == pytest.approx(-10.0)
    assert sm.geospatial_lat_max == pytest.approx(10.0)
    assert sm.geospatial_lon_min == pytest.approx(100.0)
    assert sm.geospatial_lon_max == pytest.approx(110.0)
    assert sm.time_coverage_start is not None
    assert sm.time_coverage_end is not None
    assert sm.time_coverage_start_year == 2020
    assert "2020-01-01" in sm.time_coverage_start
    assert "2020-06-01" in sm.time_coverage_end
    assert sm.s3_uri == "s3://bucket/file.nc"


def test_returns_none_fields_for_missing_coordinates(extractor):
    ds = xarray.Dataset()
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/empty.nc"))

    assert result.status == "succeeded"
    sm = result.structured_metadata
    assert sm.geospatial_lat_min is None
    assert sm.geospatial_lat_max is None
    assert sm.geospatial_lon_min is None
    assert sm.geospatial_lon_max is None
    assert sm.time_coverage_start is None
    assert sm.time_coverage_end is None


def test_extracts_crs_from_crs_wkt_attribute(extractor):
    ds = make_dataset(
        lat=[0.0],
        grid_mapping_var="crs",
        grid_mapping_var_attrs={"crs_wkt": "PROJCS[...]"},
    )
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/crs.nc"))

    assert result.structured_metadata.crs == "PROJCS[...]"


def test_extracts_crs_from_grid_mapping_name_when_no_crs_wkt(extractor):
    ds = make_dataset(
        lat=[0.0],
        grid_mapping_var="crs",
        grid_mapping_var_attrs={"grid_mapping_name": "transverse_mercator"},
    )
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/crs.nc"))

    assert result.structured_metadata.crs == "transverse_mercator"


def test_extracts_crs_from_global_attrs_when_no_grid_mapping(extractor):
    ds = make_dataset(lat=[0.0], global_attrs={"crs": "EPSG:4326"})
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/crs.nc"))

    assert result.structured_metadata.crs == "EPSG:4326"


def test_returns_failed_status_when_extraction_raises(extractor):
    class BrokenHandle:
        object_ref = ObjectReference(bucket="bucket", key="broken.nc", version_id="v1")
        file_format = None

        @property
        def s3_uri(self):
            return self.object_ref.as_uri()

        @property
        def ds(self):
            raise RuntimeError("broken")

        def cleanup(self):
            pass

    result = extractor.extract(BrokenHandle())

    assert result.status == "failed"
    assert result.error is not None
    assert result.structured_metadata is None
    assert result.unstructured_metadata is None


# --- Unstructured metadata extraction ---


def test_unstructured_metadata_contains_global_attrs_variables_coordinates(extractor):
    ds = make_dataset(lat=[-5.0, 5.0], global_attrs={"title": "Test dataset"})
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    um = result.unstructured_metadata
    assert "global_attrs" in um
    assert um["global_attrs"]["title"] == "Test dataset"
    assert "variables" in um
    assert "coordinates" in um
    assert "latitude" in um["coordinates"]


def test_unstructured_metadata_is_json_serialisable(extractor):
    ds = make_dataset(lat=[-5.0, 5.0], global_attrs={"title": "Test"})
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    # Should not raise
    json.dumps(result.unstructured_metadata)


def test_unstructured_metadata_with_numpy_bool_attribute_is_json_serialisable(
    extractor,
):
    """numpy.bool_ attributes (common in CF-NetCDF files) must be sanitised to native bool."""
    ds = make_dataset(
        lat=[-5.0, 5.0], global_attrs={"is_calibrated": numpy.bool_(True)}
    )
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    # Should not raise — numpy.bool_ is not natively JSON-serialisable
    serialised = json.dumps(result.unstructured_metadata)
    data = json.loads(serialised)
    assert data["global_attrs"]["is_calibrated"] is True


def test_unstructured_metadata_with_surrogate_string_attribute_is_sanitized(extractor):
    ds = make_dataset(
        lat=[-5.0, 5.0],
        global_attrs={"title": f"{chr(0xDCFF)}bad"},
    )
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    assert result.status == "succeeded"
    assert result.unstructured_metadata["global_attrs"]["title"] == "\\udcffbad"
    json.dumps(result.unstructured_metadata)


def test_extracts_facility_as_second_key_segment(extractor):
    ds = xarray.Dataset()
    result = extractor.extract(
        StubHandle(ds, s3_uri="s3://imos-data/IMOS/ANMN/NSW/file.nc")
    )

    assert result.structured_metadata.facility == "ANMN"


def test_facility_is_unknown_for_short_uri(extractor):
    ds = xarray.Dataset()
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/file.nc"))

    assert result.structured_metadata.facility == "UNKNOWN"


def test_facility_is_unknown_for_single_segment_key(extractor):
    """A file one level deep must not have its filename treated as a facility."""
    ds = xarray.Dataset()
    result = extractor.extract(StubHandle(ds, s3_uri="s3://bucket/IMOS/file.nc"))

    assert result.structured_metadata.facility == "UNKNOWN"


def test_file_format_propagates_to_structured_and_unstructured(extractor):
    ds = make_dataset(lat=[-5.0, 5.0])
    result = extractor.extract(StubHandle(ds, file_format="NETCDF4"))

    assert result.structured_metadata.file_format == "NETCDF4"
    assert result.unstructured_metadata["file_format"] == "NETCDF4"
