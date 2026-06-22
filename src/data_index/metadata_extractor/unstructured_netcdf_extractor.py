import typing

import pydantic
import xarray

from data_index._collection import derive_facility
from data_index.metadata_extractor._sanitize import _serialize_with_orjson
from data_index.protocols import ObjectReference, RawExtractionResult, XarrayHandle
from data_index.structured_metadata import StructuredMetadata


class UnstructuedNetCDFExtractor(pydantic.BaseModel):
    """MetadataExtractor implementation for CF-compliant NetCDF files using xarray."""

    type: typing.Literal["unstructured_netcdf_extractor"] = pydantic.Field(
        default="unstructured_netcdf_extractor"
    )

    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        try:
            structured = self._extract_structured(
                ds=handle.ds,
                object_ref=handle.object_ref,
                file_format=handle.file_format,
            )
            unstructured = self._extract_unstructured(
                ds=handle.ds,
                file_format=handle.file_format,
            )
            return RawExtractionResult(
                object_ref=handle.object_ref,
                structured_metadata=structured,
                unstructured_metadata=unstructured,
                status="succeeded",
            )
        except Exception as exc:
            return RawExtractionResult(
                object_ref=handle.object_ref,
                structured_metadata=None,
                unstructured_metadata=None,
                status="failed",
                error=str(exc),
            )

    def _extract_structured(
        self,
        ds: xarray.Dataset,
        object_ref: ObjectReference,
        file_format: str | None = None,
    ) -> StructuredMetadata:
        """
        Dummy class that
        """
        # TODO: Add the CF mapping from global attributes
        # TODO: Get Marty/core team to check over first pass structured metadata
        # TODO: Also include:
        # `Conventions`
        # `Site`, `Platform` and `Deployment`
        # `keywords`
        return StructuredMetadata(
            bucket=object_ref.bucket,
            key=object_ref.key,
            version_id=object_ref.version_id,
            facility=derive_facility(object_ref.key),
            file_format=file_format,
            geospatial_lat_min=None,
            geospatial_lat_max=None,
            geospatial_lon_min=None,
            geospatial_lon_max=None,
            time_coverage_start=None,
            time_coverage_end=None,
            crs=None,
        )

    def _extract_unstructured(
        self,
        ds: xarray.Dataset,
        file_format: str | None = None,
    ) -> dict:
        unstructured = {
            "file_format": file_format,
            "global_attrs": dict(ds.attrs),
            "variables": {
                name: {"attrs": dict(var.attrs), "dims": list(var.dims)}
                for name, var in ds.data_vars.items()
            },
            "coordinates": {
                name: {"attrs": dict(coord.attrs), "dims": list(coord.dims)}
                for name, coord in ds.coords.items()
            },
        }
        return _serialize_with_orjson(data=unstructured)
