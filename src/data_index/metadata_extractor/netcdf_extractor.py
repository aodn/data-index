import re
import typing

import pydantic
import xarray

from data_index._collection import derive_facility
from data_index.metadata_extractor._sanitize import _serialize_with_orjson
from data_index.protocols import ObjectReference, RawExtractionResult, XarrayHandle
from data_index.structured_metadata import StructuredMetadata


class NetCDFExtractor(pydantic.BaseModel):
    """MetadataExtractor implementation for CF-compliant NetCDF files using xarray."""

    type: typing.Literal["net_cdf_extractor"] = pydantic.Field(
        default="net_cdf_extractor"
    )

    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        try:
            file_format = handle.file_format
            structured = self._extract_structured(
                handle.ds, handle.object_ref, file_format
            )
            unstructured = self._extract_unstructured(handle.ds, file_format)
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

    @staticmethod
    def _extract_year(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"\b(\d{4})\b", value)
        return int(match.group(1)) if match else None

    def _extract_structured(
        self, ds: xarray.Dataset, object_ref: ObjectReference, file_format: str | None
    ) -> StructuredMetadata:
        lat_coord = next(
            (c for c in ds.coords if c in ("LATITUDE", "latitude", "lat")), None
        )
        lon_coord = next(
            (c for c in ds.coords if c in ("LONGITUDE", "longitude", "lon")), None
        )
        time_coord = next((c for c in ds.coords if c in ("TIME", "time")), None)

        lat_min = lat_max = lon_min = lon_max = None
        time_min = time_max = None

        if lat_coord:
            vals = ds.coords[lat_coord]
            lat_min, lat_max = float(vals.min()), float(vals.max())

        if lon_coord:
            vals = ds.coords[lon_coord]
            lon_min, lon_max = float(vals.min()), float(vals.max())

        if time_coord:
            vals = ds.coords[time_coord]
            time_min, time_max = str(vals.min().values), str(vals.max().values)

        crs = None
        for var in ds.data_vars.values():
            gm_name = var.attrs.get("grid_mapping")
            if gm_name and gm_name in ds:
                gm_var = ds[gm_name]
                crs = gm_var.attrs.get("crs_wkt") or gm_var.attrs.get(
                    "grid_mapping_name"
                )
                break
        if crs is None:
            crs = ds.attrs.get("crs")

        return StructuredMetadata(
            bucket=object_ref.bucket,
            key=object_ref.key,
            version_id=object_ref.version_id,
            geospatial_lat_min=lat_min,
            geospatial_lat_max=lat_max,
            geospatial_lon_min=lon_min,
            geospatial_lon_max=lon_max,
            time_coverage_start=time_min,
            time_coverage_start_year=self._extract_year(time_min),
            time_coverage_end=time_max,
            crs=crs,
            file_format=file_format,
            facility=derive_facility(object_ref.key),
        )

    def _extract_unstructured(
        self, ds: xarray.Dataset, file_format: str | None
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
