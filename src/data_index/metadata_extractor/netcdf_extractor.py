import xarray
import numpy
from data_index.protocols import RawExtractionResult, StructuredMetadata, XarrayHandle
from data_index.metadata_extractor._sanitize import _sanitize_for_json, _serialize_with_orjson


class NetCDFExtractor:
    """MetadataExtractor implementation for CF-compliant NetCDF files using xarray."""

    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        try:
            file_format = handle.file_format
            structured = self._extract_structured(handle.ds, handle.s3_uri, file_format)
            unstructured = self._extract_unstructured(handle.ds, file_format)
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

    def _extract_structured(self, ds: xarray.Dataset, s3_uri: str, file_format: str | None) -> StructuredMetadata:
        lat_coord = next((c for c in ds.coords if c in ("LATITUDE", "latitude", "lat")), None)
        lon_coord = next((c for c in ds.coords if c in ("LONGITUDE", "longitude", "lon")), None)
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
                crs = gm_var.attrs.get("crs_wkt") or gm_var.attrs.get("grid_mapping_name")
                break
        if crs is None:
            crs = ds.attrs.get("crs")

        return StructuredMetadata(
            s3_uri=s3_uri,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            time_min=time_min,
            time_max=time_max,
            crs=crs,
            file_format=file_format,
        )

    def _extract_unstructured(self, ds: xarray.Dataset, file_format: str | None) -> dict:
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