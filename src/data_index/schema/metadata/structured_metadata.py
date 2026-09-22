import dataclasses
import decimal
import math
import types
import typing
from functools import cache

from .base_metadata import BaseMetadata


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class StructuredMetadata(BaseMetadata):
    """Structured metadata row schema and backend schema converters.

    `StructuredMetadata` is source-of-truth for Polars, PyArrow, and PyIceberg
    schema generation.
    """

    # Upgrade the schema version when changing the schema
    SCHEMA_VERSION: typing.ClassVar[int] = 6
    schema_version: int = dataclasses.field(default=SCHEMA_VERSION)

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
    variable_schema: dict[str, str] | None = None
    coordinate_schema: dict[str, str] | None = None
    dimension_sizes: dict[str, int] | None = None
    standard_names: dict[str, str] | None = None

    @staticmethod
    def _is_float_annotation(annotation: typing.Any) -> bool:
        if annotation is float:
            return True
        origin = typing.get_origin(annotation)
        if origin in (typing.Union, types.UnionType):
            args = typing.get_args(annotation)
            non_none_args = tuple(arg for arg in args if arg is not type(None))
            return len(non_none_args) == 1 and non_none_args[0] is float
        return False

    @classmethod
    @cache
    def _conversion_plan(cls) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return cached field groups for DynamoDB conversion.

        This runs expensive type-hint inspection exactly once per process.
        The row serializer is a hot path (one metadata object per file), so
        we precompute which fields need float->Decimal coercion.
        """
        structured_hints = typing.get_type_hints(cls, include_extras=True)
        float_fields = tuple(
            field.name
            for field in dataclasses.fields(cls)
            if cls._is_float_annotation(structured_hints[field.name])
        )
        passthrough_fields = tuple(
            field.name
            for field in dataclasses.fields(cls)
            if field.name not in float_fields
        )
        return passthrough_fields, float_fields

    @property
    def dynamodb_item(self) -> dict[str, typing.Any]:
        """Return a DynamoDB-compatible row dict with Decimal numeric fields.

        Performance note:
        We intentionally avoid `dataclasses.asdict()` because it performs a
        recursive deep copy on every row. Metadata sink writes run per-row, so
        shallow field reads + cached conversion plans are materially cheaper.
        """
        passthrough_fields, float_fields = self._conversion_plan()
        source = self.__dict__
        row_item = {field_name: source[field_name] for field_name in passthrough_fields}
        for field_name in float_fields:
            value = source[field_name]
            if value is None:
                row_item[field_name] = None
                continue
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError(
                        f"Structured field '{field_name}' has non-finite float value: {value}"
                    )
                row_item[field_name] = decimal.Decimal(str(value))
            else:
                row_item[field_name] = value
        return row_item
