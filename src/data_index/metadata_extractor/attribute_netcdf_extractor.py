import pydantic
import xarray

from data_index.metadata_extractor._sanitize import (
    _serialize_with_orjson,
)
from data_index.protocols import RawExtractionResult, XarrayHandle
from data_index.structured_metadata import StructuredMetadata


class AttributeNetCDFExtractor(pydantic.BaseModel):
    """MetadataExtractor implementation for CF-compliant NetCDF files using xarray."""

    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        try:
            structured = self._extract_structured(
                ds=handle.ds,
                s3_uri=handle.s3_uri,
                file_format=handle.file_format,
            )
            unstructured = self._extract_unstructured(
                ds=handle.ds,
                file_format=handle.file_format,
            )
            return RawExtractionResult(
                s3_uri=handle.s3_uri,
                structured_metadata=structured,
                unstructured_metadata=unstructured,
                status="succeeded",
            )
        except Exception as exc:
            return RawExtractionResult(
                s3_uri=handle.s3_uri,
                structured_metadata=None,
                unstructured_metadata=None,
                status="failed",
                error=str(exc),
            )

    def _extract_structured(
        self,
        ds: xarray.Dataset,
        s3_uri: str,
        file_format: str | None = None,
    ) -> StructuredMetadata:
        """
        Structured data extraction from attributes
        """
        metadata_kwargs = {
            "s3_uri": s3_uri,
            "file_format": file_format,
        }

        attributes_map: dict[str, tuple[str, type]] = {
            # Temporal Spatial
            "lat_min": ("geospatial_lat_min", float),
            "lat_max": ("geospatial_lat_max", float),
            "lon_min": ("geospatial_lon_min", float),
            "lon_max": ("geospatial_lon_max", float),
            "time_min": ("time_coverage_start", str),
            "time_max": ("time_coverage_end", str),
            # Keywords
            "keywords": ("keywords", str),
            "conventions": ("Conventions", str),
            "file_version": ("file_version", str),
            "file_version_quality_control": ("file_version_quality_control", str),
            "metadata_uuid": ("metadata_uuid", str),
            # Site Platform Deployment
            "platform_code": ("platform_code", str),
            "site_code": ("site_code", str),
            "deployment_code": ("deployment_code", str),
            # Instrumentation
            "instrument": ("instrument", str),
        }

        errors = {}

        # Convert attribute
        for attribute, (key, _type) in attributes_map.items():
            val = ds.attrs.get(key)
            try:
                metadata_kwargs[attribute] = _type(val) if val is not None else None
            except (ValueError, TypeError) as e:
                metadata_kwargs[attribute] = None
                errors[attribute] = e

        return StructuredMetadata(**metadata_kwargs)

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
