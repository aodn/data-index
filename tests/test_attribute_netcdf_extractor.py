import xarray

from data_index.metadata_extractor.attribute_netcdf_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.protocols import ObjectReference


class StubHandle:
    def __init__(
        self,
        ds: xarray.Dataset,
        s3_uri: str = "s3://imos-data/IMOS/ANMN/NSW/file.nc",
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


def test_extract_supports_aliases_for_structured_scalar_attributes():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset(
        attrs={
            "conventions": "CF-1.8",
            "featureType": "trajectory",
            "instrumentSerialNumber": "INS-001",
        }
    )

    result = extractor.extract(StubHandle(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.conventions == "CF-1.8"
    assert result.structured_metadata.feature_type == "trajectory"
    assert result.structured_metadata.instrument_serial_number == "INS-001"


def test_extract_prioritizes_exact_alias_order_over_normalized_fallback():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset(
        attrs={
            "Conventions": "CF-UPPER",
            "conventions": "CF-lower",
            "instrument-serial-number": "INS-fallback",
        }
    )

    result = extractor.extract(StubHandle(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.conventions == "CF-UPPER"
    assert result.structured_metadata.instrument_serial_number == "INS-fallback"


def test_extract_derives_dimensions_variables_and_standard_names():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset(
        data_vars={
            "temp": xarray.DataArray(
                [[1.0, 2.0]],
                dims=("time", "lat"),
                attrs={"standard_name": "sea_water_temperature"},
            ),
            "salinity": xarray.DataArray(
                [[35.0, 35.1]],
                dims=("time", "lat"),
                attrs={"standard_name": "sea_water_salinity"},
            ),
        },
        coords={
            "time": xarray.DataArray(
                [0], dims=("time",), attrs={"standard_name": "time"}
            ),
            "lat": xarray.DataArray(
                [10.0, 11.0], dims=("lat",), attrs={"standard_name": "latitude"}
            ),
        },
    )

    result = extractor.extract(StubHandle(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.dimensions == ["lat", "time"]
    assert result.structured_metadata.variables == ["salinity", "temp"]
    assert result.structured_metadata.standard_names == [
        "latitude",
        "sea_water_salinity",
        "sea_water_temperature",
        "time",
    ]


def test_extract_sets_derived_lists_to_none_when_empty():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset()

    result = extractor.extract(StubHandle(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.dimensions is None
    assert result.structured_metadata.variables is None
    assert result.structured_metadata.standard_names is None


def test_extract_succeeds_with_surrogate_string_in_global_attrs():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset(attrs={"title": f"{chr(0xDCFF)}bad"})

    result = extractor.extract(StubHandle(ds))

    assert result.status == "succeeded"
    assert result.unstructured_metadata is not None
    assert result.unstructured_metadata["global_attrs"]["title"] == "\\udcffbad"
