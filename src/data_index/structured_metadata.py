import dataclasses
import types
import typing

import polars
import pyarrow


@dataclasses.dataclass
class StructuredMetadata:
    s3_uri: str
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    time_min: str | None = None
    time_max: str | None = None
    crs: str | None = None
    keywords: str | None = None
    conventions: str | None = None
    file_version: str | None = None
    file_version_quality_control: str | None = None
    metadata_uuid: str | None = None
    platform_code: str | None = None
    site_code: str | None = None
    deployment_code: str | None = None
    instrument: str | None = None
    file_format: str | None = None
    collection: str | None = None

    @classmethod
    def _unpack_field(cls, field: dataclasses.Field):
        """
        Helper function to unpack the a dataclass field.
        """

        origin = typing.get_origin(tp=field.type)

        # Parse union types
        if origin in (typing.Union, types.UnionType):
            # Get the unioned types
            args = typing.get_args(field.type)

            # If None in the union then it is nullable
            if type(None) in args:
                is_nullable = True

            # Otherwise it is an invalid input
            else:
                raise ValueError(
                    f"Cannot generate schema for union of two types (excluding None): {origin}"
                )

            # Check what the remaining type is
            remaining_types = [t for t in args if t is not type(None)]

            return {
                "name": field.name,
                "type": remaining_types[0] if remaining_types else type(None),
                "nullable": is_nullable,
            }

        # Parse non-union type
        else:
            return {
                "name": field.name,
                "type": field.type,
                "nullable": False,
            }

    @classmethod
    def _unpack_fields(cls) -> dict:
        """
        Helper function to unpack the fields of a dataclass
        """

        return [
            cls._unpack_field(field=field)
            for field in dataclasses.fields(class_or_instance=cls)
        ]

    @classmethod
    def as_polars_schema(cls) -> polars.Schema:

        _POLARS_TYPE_MAP = {
            str: polars.String,
            float: polars.Float64,
            int: polars.Int64,
            bool: polars.Boolean,
        }
        return polars.Schema(
            schema={
                field["name"]: _POLARS_TYPE_MAP[field["type"]]
                for field in cls._unpack_fields()
            }
        )

    @classmethod
    def as_pyarrow_schema(cls) -> pyarrow.Schema:
        _PYARROW_TYPE_MAP = {
            str: pyarrow.string(),
            float: pyarrow.float64(),
            int: pyarrow.int64(),
            bool: pyarrow.bool_(),
        }
        return pyarrow.schema(
            fields=[
                pyarrow.field(
                    field["name"],
                    type=_PYARROW_TYPE_MAP[field["type"]],
                    nullable=field["nullable"],
                )
                for field in cls._unpack_fields()
            ]
        )
