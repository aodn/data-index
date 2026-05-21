import xarray
import numpy
from data_index.protocols import RawExtractionResult, StructuredMetadata


class NetCDFExtractor:
    """MetadataExtractor implementation for CF-compliant NetCDF files using xarray."""

    def extract(self, ds: xarray.Dataset, s3_uri: str) -> RawExtractionResult:
        try:
            structured = self._extract_structured(ds, s3_uri)
            unstructured = self._extract_unstructured(ds)
            return RawExtractionResult(
                s3_uri=s3_uri,
                structured_metadata=structured,
                unstructured_metadata=unstructured,
                status="succeeded",
            )
        except Exception as exc:
            return RawExtractionResult(
                s3_uri=s3_uri,
                structured_metadata=None,
                unstructured_metadata=None,
                status="failed",
                error=str(exc),
            )

    def _extract_structured(self, ds: xarray.Dataset, s3_uri: str) -> StructuredMetadata:
        lat_coord = next((c for c in ds.coords if c in ("LATITUDE", "latitude", "lat")), None)
        lon_coord = next((c for c in ds.coords if c in ("LONGITUDE", "longitude", "lon")), None)
        time_coord = next((c for c in ds.coords if c in ("TIME", "time")), None)

        lat_min = lat_max = lon_min = lon_max = None
        time_min = time_max = None

        if lat_coord:
            vals = ds.coords[lat_coord].values
            lat_min, lat_max = float(vals.min()), float(vals.max())

        if lon_coord:
            vals = ds.coords[lon_coord].values
            lon_min, lon_max = float(vals.min()), float(vals.max())

        if time_coord:
            vals = ds.coords[time_coord].values
            time_min, time_max = str(vals.min()), str(vals.max())

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
        )

    def _extract_unstructured(self, ds: xarray.Dataset) -> dict:
        unstructured = {
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
        return self._sanitize_for_json(unstructured)

    def _sanitize_for_json(self, data):
        """Recursively convert numpy/xarray types to native Python primitives."""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._sanitize_for_json(v) for v in data]
        elif isinstance(data, (numpy.integer, numpy.int64, numpy.int32)):
            return int(data)
        elif isinstance(data, (numpy.floating, numpy.float64, numpy.float32)):
            return float(data)
        elif isinstance(data, numpy.ndarray):
            return data.tolist()
        # Handle cases where attributes might be byte-strings
        elif isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return data