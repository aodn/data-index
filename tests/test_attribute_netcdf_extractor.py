import json

import xarray

from data_index.metadata_extractor.attribute_netcdf_extractor import (
    AttributeNetCDFExtractor,
)
from data_index.protocols import ObjectReference


class StubXarrayHandle:
    def __init__(self, ds: xarray.Dataset):
        self._ds = ds
        self.file_format = "NETCDF4"

    @property
    def ds(self) -> xarray.Dataset:
        return self._ds


def _object_reference(
    ds: xarray.Dataset = xarray.Dataset(
        attrs={
            "conventions": "CF-1.8",
            "featureType": "trajectory",
            "instrumentSerialNumber": "INS-001",
        }
    ),
) -> ObjectReference:
    return ObjectReference(
        bucket="test",
        key="IMOS/file.nc",
        version_id="0",
        size=32,
        xarray_handle=None,
    ).with_xarray_handle(StubXarrayHandle(ds=ds))


def test_extract_supports_aliases_for_structured_scalar_attributes():
    extractor = AttributeNetCDFExtractor()

    result = extractor.extract(_object_reference())

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

    result = extractor.extract(_object_reference(ds))

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

    result = extractor.extract(_object_reference(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.dimensions == ["lat", "time"]
    assert result.structured_metadata.variables == [
        "salinity",
        "temp",
    ]
    assert result.structured_metadata.standard_names == [
        "latitude",
        "sea_water_salinity",
        "sea_water_temperature",
        "time",
    ]


def test_extract_sets_derived_lists_to_none_when_empty():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset()

    result = extractor.extract(_object_reference(ds))

    assert result.status == "succeeded"
    assert result.structured_metadata is not None
    assert result.structured_metadata.dimensions is None
    assert result.structured_metadata.variables is None
    assert result.structured_metadata.standard_names is None


def test_extract_succeeds_with_surrogate_string_in_global_attrs():
    extractor = AttributeNetCDFExtractor()
    ds = xarray.Dataset(attrs={"title": f"{chr(0xDCFF)}bad"})

    extraction_result = extractor.extract(_object_reference(ds))

    assert extraction_result.status == "succeeded"
    assert extraction_result.unstructured_metadata is not None
    metadata = json.loads(extraction_result.unstructured_metadata.metadata)
    assert metadata["global_attrs"]["title"] == "\\udcffbad"
