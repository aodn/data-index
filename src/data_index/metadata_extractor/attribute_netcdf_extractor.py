import re
import typing

import pydantic
import xarray

from data_index._collection import derive_facility
from data_index.metadata_extractor._sanitize import _serialize_with_orjson
from data_index.protocols import ExtractionResult, ObjectReference
from data_index.schema.metadata import StructuredMetadata, UnstructuredMetadata


class AttributeNetCDFExtractor(pydantic.BaseModel):
    """Metadata extractor for CF-compliant NetCDF datasets via xarray."""

    type: typing.Literal["attribute_netcdf_extractor"] = pydantic.Field(
        default="attribute_netcdf_extractor"
    )

    def extract(self, object_reference: ObjectReference) -> ExtractionResult:
        """Extract structured and unstructured metadata for one handle.

        :param handle: Dataset handle to read from.
        :returns: Extraction result with success/failure status.
        """

        try:
            structured_metadata = self._extract_structured(
                object_reference=object_reference,
            )
            unstructured_metadata = self._extract_unstructured(
                object_reference=object_reference
            )
            return ExtractionResult(
                structured_metadata=structured_metadata,
                unstructured_metadata=unstructured_metadata,
                status="succeeded",
            )
        except Exception as e:
            return ExtractionResult(
                structured_metadata=structured_metadata,
                unstructured_metadata=unstructured_metadata,
                status="failed",
                error=str(e),
            )

    @staticmethod
    def _extract_year(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"\b(\d{4})\b", value)
        return int(match.group(1)) if match else None

    def _extract_structured(
        self,
        object_reference: ObjectReference,
    ) -> StructuredMetadata:
        """Build structured metadata row from global attributes and dataset structure.

        :param ds: Open xarray dataset.
        :param object_ref: Source object identity for the row.
        :param file_format: File format derived from magic bytes.
        :returns: Structured metadata row.
        """

        # TODO:
        # For dimensions, capture the sizes of the dimensions not just the names

        metadata = {
            "hash": object_reference.hash,
            "bucket": object_reference.bucket,
            "key": object_reference.key,
            "version_id": object_reference.version_id,
            "facility": derive_facility(object_reference.key),
            "file_format": object_reference.xarray_handle.file_format,
        }

        attributes_map: dict[str, tuple[list[str], type]] = {
            # Temporal Spatial
            "geospatial_lat_min": (["geospatial_lat_min"], float),
            "geospatial_lat_max": (["geospatial_lat_max"], float),
            "geospatial_lon_min": (["geospatial_lon_min"], float),
            "geospatial_lon_max": (["geospatial_lon_max"], float),
            "geospatial_vertical_min": (["geospatial_vertical_min"], float),
            "geospatial_vertical_max": (["geospatial_vertical_max"], float),
            "geospatial_vertical_positive": (["geospatial_vertical_positive"], str),
            "time_coverage_start": (["time_coverage_start"], str),
            "time_coverage_end": (["time_coverage_end"], str),
            "date_created": (["date_created"], str),
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
            "instrument_nominal_depth": (["instrument_nominal_depth"], float),
            "feature_type": (["featureType", "feature_type"], str),
            "instrument_serial_number": (
                ["instrument_serial_number", "instrumentSerialNumber"],
                str,
            ),
        }

        errors = {}
        ds = object_reference.xarray_handle.ds

        # Convert attribute
        for attribute, (aliases, _type) in attributes_map.items():
            resolved_key = self._resolve_attribute_key(ds.attrs, aliases)
            val = ds.attrs.get(resolved_key) if resolved_key is not None else None
            try:
                metadata[attribute] = _type(val) if val is not None else None
            except (ValueError, TypeError) as e:
                metadata[attribute] = None
                errors[attribute] = e

        metadata["dimensions"] = self._sorted_or_none(ds.dims)
        metadata["variables"] = self._sorted_or_none(ds.data_vars)
        metadata["standard_names"] = self._extract_standard_names(ds)

        return StructuredMetadata(**metadata)

    def _extract_unstructured(
        self,
        object_reference: ObjectReference,
    ) -> UnstructuredMetadata:
        """Build unstructured metadata payload from dataset contents.

        :param ds: Open xarray dataset.
        :param file_format: File format derived from magic bytes.
        :returns: JSON-serializable unstructured metadata dict.
        """

        ds = object_reference.xarray_handle.ds

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

        return UnstructuredMetadata(
            bucket=object_reference.bucket,
            key=object_reference.key,
            version_id=object_reference.version_id,
            hash=object_reference.hash,
            metadata=_serialize_with_orjson(data=unstructured),
            file_format=object_reference.xarray_handle.file_format,
            facility=derive_facility(object_reference.key),
        )

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
