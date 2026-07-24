import json
import re
import typing

import pydantic
import xarray

import data_index.protocols
import data_index.schema.metadata
from data_index._collection import derive_facility
from data_index.metadata_extractor._sanitize import _serialize_with_orjson

FrozenDict = type(xarray.Dataset().sizes)


class AttributeNetCDFExtractor(pydantic.BaseModel):
    """Metadata extractor for CF-compliant NetCDF datasets via xarray."""

    type: typing.Literal["attribute_netcdf_extractor"] = pydantic.Field(
        default="attribute_netcdf_extractor"
    )

    @classmethod
    def extract(
        cls, staged_object: data_index.protocols.StagedObject
    ) -> data_index.protocols.ExtractedObject | data_index.protocols.DeadLetter:
        """Extract structured and unstructured metadata for one handle.

        :param handle: Dataset handle to read from.
        :returns: Extraction result with success/failure status.
        """
        structured_metadata = None
        unstructured_metadata = None
        try:
            structured_metadata = cls._extract_structured(
                staged_object=staged_object,
            )
            unstructured_metadata = cls._extract_unstructured(
                staged_object=staged_object
            )
            return data_index.protocols.ExtractedObject(
                object_reference=staged_object.object_reference,
                extraction_result=data_index.protocols.ExtractionResult(
                    structured_metadata=structured_metadata,
                    unstructured_metadata=unstructured_metadata,
                ),
            )
        except Exception as e:
            return data_index.protocols.DeadLetter.from_object_reference(
                object_reference=staged_object.object_reference, error=str(e)
            )

    @staticmethod
    def _extract_year(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"\b(\d{4})\b", value)
        return int(match.group(1)) if match else None

    @classmethod
    def _extract_structured(
        cls,
        staged_object: data_index.protocols.StagedObject,
    ) -> data_index.schema.metadata.StructuredMetadata:
        """Build structured metadata row from global attributes and dataset structure.

        :param ds: Open xarray dataset.
        :param object_ref: Source object identity for the row.
        :param file_format: File format derived from magic bytes.
        :returns: Structured metadata row.
        """

        # TODO:
        # For dimensions, capture the sizes of the dimensions not just the names

        metadata = {
            "hash": staged_object.object_reference.hash,
            "bucket": staged_object.object_reference.bucket,
            "key": staged_object.object_reference.key,
            "version_id": staged_object.object_reference.version_id,
            "facility": derive_facility(staged_object.object_reference.key),
            "file_format": staged_object.xarray_handle.file_format,
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
        ds = staged_object.xarray_handle.ds

        # Convert attribute
        for attribute, (aliases, _type) in attributes_map.items():
            resolved_key = cls._resolve_attribute_key(ds.attrs, aliases)
            val = ds.attrs.get(resolved_key) if resolved_key is not None else None
            try:
                metadata[attribute] = _type(val) if val is not None else None
            except (ValueError, TypeError) as e:
                metadata[attribute] = None
                errors[attribute] = e

        # Extract netCDF Shape Metadata
        variable_schema = {
            variable: str(ds.variables[variable].dtype)
            for variable in sorted(ds.data_vars)
        } or None
        coordinate_schema = {
            coordinate: str(ds.coords[coordinate].dtype)
            for coordinate in sorted(ds.coords)
        } or None
        dimension_sizes = {
            dimension: ds.sizes[dimension] for dimension in sorted(ds.sizes)
        } or None
        standard_names = {
            variable: ds.variables[variable].attrs.get("standard_name")
            for variable in sorted(ds.variables)
            if ds.variables[variable].attrs.get("standard_name")
        } or None
        metadata["variable_schema"] = variable_schema
        metadata["coordinate_schema"] = coordinate_schema
        metadata["dimension_sizes"] = dimension_sizes
        metadata["standard_names"] = standard_names

        return data_index.schema.metadata.StructuredMetadata(**metadata)

    @classmethod
    def _extract_unstructured(
        cls,
        staged_object: data_index.protocols.StagedObject,
    ) -> data_index.schema.metadata.UnstructuredMetadata:
        """Build unstructured metadata payload from dataset contents.

        :param ds: Open xarray dataset.
        :param file_format: File format derived from magic bytes.
        :returns: JSON-serializable unstructured metadata dict.
        """

        ds = staged_object.xarray_handle.ds

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

        return data_index.schema.metadata.UnstructuredMetadata(
            bucket=staged_object.object_reference.bucket,
            key=staged_object.object_reference.key,
            version_id=staged_object.object_reference.version_id,
            hash=staged_object.object_reference.hash,
            metadata=json.dumps(
                obj=_serialize_with_orjson(data=unstructured),
                indent=None,
            ),
            file_format=staged_object.xarray_handle.file_format,
            facility=derive_facility(staged_object.object_reference.key),
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
