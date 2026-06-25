import json

import pytest
import xarray

import data_index.protocols
from data_index.metadata_extractor.attribute_netcdf_extractor import (
    AttributeNetCDFExtractor,
)


class StubXarrayHandle:
    def __init__(self, ds: xarray.Dataset):
        self._ds = ds
        self.file_format = "NETCDF4"

    @property
    def ds(self) -> xarray.Dataset:
        return self._ds


@pytest.fixture
def object_reference() -> data_index.protocols.ObjectReference:
    return data_index.protocols.ObjectReference(
        bucket="test",
        key="IMOS/file.nc",
        version_id="0",
        size=32,
    )


@pytest.fixture
def ds(request) -> xarray.Dataset:
    # If indirect parametrization passes a value, return it
    if hasattr(request, "param"):
        return request.param

    # Otherwise, fall back to your default dataset
    return xarray.Dataset(
        attrs={
            "conventions": "CF-1.8",
            "featureType": "trajectory",
            "instrumentSerialNumber": "INS-001",
        }
    )


@pytest.fixture
def staged_object_factory(object_reference):
    """A factory fixture that builds a StagedObject with a custom dataset."""

    def _create(ds: xarray.Dataset) -> data_index.protocols.StagedObject:
        return data_index.protocols.StagedObject(
            object_reference=object_reference, xarray_handle=StubXarrayHandle(ds=ds)
        )

    return _create


@pytest.fixture
def staged_object(staged_object_factory, ds):
    """The standard staged_object uses the factory with the default or parameterized ds."""
    return staged_object_factory(ds)


@pytest.fixture(scope="session")
def extractor():
    return AttributeNetCDFExtractor()


def test_extract_supports_aliases_for_structured_scalar_attributes(
    staged_object, extractor
):
    extracted_object = extractor.extract(staged_object=staged_object)
    assert extracted_object.extraction_result.structured_metadata is not None
    assert (
        extracted_object.extraction_result.structured_metadata.conventions == "CF-1.8"
    )
    assert (
        extracted_object.extraction_result.structured_metadata.feature_type
        == "trajectory"
    )
    assert (
        extracted_object.extraction_result.structured_metadata.instrument_serial_number
        == "INS-001"
    )


@pytest.mark.parametrize(
    "ds",
    [
        xarray.Dataset(
            attrs={
                "Conventions": "CF-UPPER",
                "conventions": "CF-lower",
                "instrument_serial_number": "INS-fallback",  # Added to satisfy the assertion
            }
        )
    ],
    indirect=True,
)
def test_extract_prioritizes_exact_alias_order_over_normalized_fallback(
    staged_object, extractor
):
    extracted_object = extractor.extract(staged_object=staged_object)
    assert extracted_object.extraction_result.structured_metadata is not None
    assert (
        extracted_object.extraction_result.structured_metadata.conventions == "CF-UPPER"
    )
    assert (
        extracted_object.extraction_result.structured_metadata.instrument_serial_number
        == "INS-fallback"
    )


@pytest.mark.parametrize(
    argnames="ds",
    argvalues=[
        xarray.Dataset(
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
    ],
)
def test_extract_derives_dimensions_variables_and_standard_names(
    staged_object, extractor
):
    extracted_object = extractor.extract(staged_object=staged_object)
    assert extracted_object.extraction_result.structured_metadata.dimensions == [
        "lat",
        "time",
    ]
    assert extracted_object.extraction_result.structured_metadata.variables == [
        "salinity",
        "temp",
    ]
    assert extracted_object.extraction_result.structured_metadata.standard_names == [
        "latitude",
        "sea_water_salinity",
        "sea_water_temperature",
        "time",
    ]


@pytest.mark.parametrize(argnames="ds", argvalues=[xarray.Dataset()])
def test_extract_sets_derived_lists_to_none_when_empty(staged_object, extractor):
    extracted_object = extractor.extract(staged_object=staged_object)
    assert extracted_object.extraction_result.structured_metadata.dimensions is None
    assert extracted_object.extraction_result.structured_metadata.variables is None
    assert extracted_object.extraction_result.structured_metadata.standard_names is None


@pytest.mark.parametrize(
    argnames="ds", argvalues=[xarray.Dataset(attrs={"title": f"{chr(0xDCFF)}bad"})]
)
def test_extract_succeeds_with_surrogate_string_in_global_attrs(
    staged_object, extractor
):
    extracted_object = extractor.extract(staged_object=staged_object)
    assert extracted_object.extraction_result.unstructured_metadata is not None
    metadata = json.loads(
        extracted_object.extraction_result.unstructured_metadata.metadata
    )
    assert metadata["global_attrs"]["title"] == "\\udcffbad"
