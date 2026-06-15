import typing

import pydantic
import xarray

from data_index.metadata_extractor._sanitize import (
    _serialize_with_orjson,
)
from data_index.protocols import RawExtractionResult, XarrayHandle
from data_index.structured_metadata import StructuredMetadata


class AttributeNetCDFExtractor(pydantic.BaseModel):
    """Metadata extractor for CF-compliant NetCDF datasets via xarray."""

    type: typing.Literal["attribute_netcdf_extractor"] = pydantic.Field(
        default="attribute_netcdf_extractor"
    )

    def extract(self, handle: XarrayHandle) -> RawExtractionResult:
        """Extract structured and unstructured metadata for one handle.

        :param handle: Dataset handle to read from.
        :returns: Extraction result with success/failure status.
        """

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
        """Build structured metadata row from global attributes and dataset structure.

        :param ds: Open xarray dataset.
        :param s3_uri: Source S3 URI for the row.
        :param file_format: File format derived from magic bytes.
        :returns: Structured metadata row.
        """

        # TODO:
        # For dimensions, capture the sizes of the dimensions not just the names

        metadata_kwargs = {
            "s3_uri": s3_uri,
            "file_format": file_format,
        }

        attributes_map: dict[str, tuple[list[str], type]] = {
            # Temporal Spatial
            "lat_min": (["geospatial_lat_min"], float),
            "lat_max": (["geospatial_lat_max"], float),
            "lon_min": (["geospatial_lon_min"], float),
            "lon_max": (["geospatial_lon_max"], float),
            "time_min": (["time_coverage_start"], str),
            "time_max": (["time_coverage_end"], str),
            # Keywords
            "keywords": (["keywords"], str),
            "conventions": (["Conventions", "conventions"], str),
            "file_version": (["file_version"], str),
            "metadata_uuid": (["metadata_uuid"], str),
            # Site Platform Deployment
            "platform_code": (["platform_code"], str),
            "site_code": (["site_code"], str),
            "deployment_code": (["deployment_code"], str),
            # Instrumentation
            "instrument": (["instrument"], str),
            "feature_type": (["featureType", "feature_type"], str),
            "instrument_serial_number": (
                ["instrument_serial_number", "instrumentSerialNumber"],
                str,
            ),
        }

        errors = {}

        # Convert attribute
        for attribute, (aliases, _type) in attributes_map.items():
            resolved_key = self._resolve_attribute_key(ds.attrs, aliases)
            val = ds.attrs.get(resolved_key) if resolved_key is not None else None
            try:
                metadata_kwargs[attribute] = _type(val) if val is not None else None
            except (ValueError, TypeError) as e:
                metadata_kwargs[attribute] = None
                errors[attribute] = e

        metadata_kwargs["dimensions"] = self._sorted_or_none(ds.dims)
        metadata_kwargs["variables"] = self._sorted_or_none(ds.data_vars)
        metadata_kwargs["standard_names"] = self._extract_standard_names(ds)

        return StructuredMetadata(**metadata_kwargs)

    @staticmethod
    def _normalize_attr_key(key: str) -> str:
        """Normalize attribute keys for discrepancy-tolerant matching.

        :param key: Raw attribute key.
        :returns: Lower-cased key with ``_`` and ``-`` removed.
        """

        return key.lower().replace("_", "").replace("-", "")

    @classmethod
    def _resolve_attribute_key(cls, attrs: dict, aliases: list[str]) -> str | None:
        """Resolve attribute key by alias priority then normalized fallback.

        :param attrs: Dataset global attributes.
        :param aliases: Ordered canonical aliases, highest priority first.
        :returns: Matched attribute key from ``attrs`` or ``None``.
        """

        for alias in aliases:
            if alias in attrs:
                return alias

        for alias in aliases:
            normalized_alias = cls._normalize_attr_key(alias)
            for key in attrs:
                if (
                    isinstance(key, str)
                    and cls._normalize_attr_key(key) == normalized_alias
                ):
                    return key
        return None

    @staticmethod
    def _sorted_or_none(values) -> list[str] | None:
        """Return sorted unique string values, or ``None`` when empty.

        :param values: Iterable-like values to normalize.
        :returns: Sorted unique strings or ``None``.
        """

        normalized = sorted({str(value) for value in values})
        return normalized or None

    @classmethod
    def _extract_standard_names(cls, ds: xarray.Dataset) -> list[str] | None:
        """Collect unique ``standard_name`` values from vars and coords.

        :param ds: Open xarray dataset.
        :returns: Sorted unique standard names or ``None``.
        """

        standard_names = set()
        for variable in list(ds.data_vars.values()) + list(ds.coords.values()):
            value = variable.attrs.get("standard_name")
            if isinstance(value, str) and value.strip():
                standard_names.add(value)
        return cls._sorted_or_none(standard_names)

    def _extract_unstructured(
        self,
        ds: xarray.Dataset,
        file_format: str | None = None,
    ) -> dict:
        """Build unstructured metadata payload from dataset contents.

        :param ds: Open xarray dataset.
        :param file_format: File format derived from magic bytes.
        :returns: JSON-serializable unstructured metadata dict.
        """

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
