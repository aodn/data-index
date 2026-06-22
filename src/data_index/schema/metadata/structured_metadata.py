import typing

from .base import Metadata


class StructuredMetadata(Metadata):
    """Structured metadata row schema and backend schema converters.

    ``StructuredMetadata`` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    # Upgrade the schema version when changing the schema
    SCHEMA_VERSION: typing.ClassVar[int] = 2
    schema_version: int = SCHEMA_VERSION

    geospatial_lat_min: float | None = None
    geospatial_lat_max: float | None = None
    geospatial_lon_min: float | None = None
    geospatial_lon_max: float | None = None
    geospatial_vertical_min: float | None = None
    geospatial_vertical_max: float | None = None
    geospatial_vertical_positive: str | None = None
    time_coverage_start: str | None = None
    time_coverage_end: str | None = None
    date_created: str | None = None
    crs: str | None = None
    keywords: str | None = None
    conventions: str | None = None
    file_version: str | None = None
    metadata_uuid: str | None = None
    platform_code: str | None = None
    site_code: str | None = None
    deployment_code: str | None = None
    instrument: str | None = None
    instrument_nominal_depth: float | None = None
    feature_type: str | None = None
    instrument_serial_number: str | None = None
    dimensions: list[str] | None = None
    variables: list[str] | None = None
    standard_names: list[str] | None = None
    file_format: str | None = None
    collection: str | None = None
