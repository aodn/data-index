import dataclasses
import types
import typing

import polars
import pyarrow
import pyiceberg.schema
import pyiceberg.types


@dataclasses.dataclass(frozen=True)
class _TypeSpec:
    """Normalized internal representation of a field type.

    :param kind: Type category, either ``"scalar"`` or ``"list"``.
    :param scalar_type: Python scalar type when ``kind == "scalar"``.
    :param item_type: Nested element type when ``kind == "list"``.
    """

    kind: typing.Literal["scalar", "list"]
    scalar_type: type | None = None
    item_type: "_TypeSpec | None" = None


@dataclasses.dataclass(frozen=True)
class _FieldSpec:
    """Resolved schema information for one dataclass field.

    :param name: Field name.
    :param type_spec: Parsed type specification.
    :param nullable: Whether the field allows ``None``.
    """

    name: str
    type_spec: _TypeSpec
    nullable: bool


@dataclasses.dataclass
class _PyIcebergIdAllocator:
    """Allocates unique nested field IDs for PyIceberg complex types.

    :param next_id: Next available ID.
    """

    next_id: int

    def next_nested_id(self) -> int:
        """Return next nested field ID and advance allocator.

        :returns: Next unique nested ID.
        """

        nested_id = self.next_id
        self.next_id += 1
        return nested_id


class Schema:
    @classmethod
    def _parse_annotation(cls, annotation) -> tuple[_TypeSpec, bool]:
        """Parse a field annotation into internal type spec + nullable flag.

        :param annotation: Field annotation from dataclass metadata.
        :returns: ``(type_spec, nullable)``.
        :raises ValueError: If annotation is unsupported or ambiguous.
        """

        origin = typing.get_origin(tp=annotation)
        if origin in (typing.Union, types.UnionType):
            args = typing.get_args(annotation)
            nullable = type(None) in args
            if not nullable:
                raise ValueError(
                    f"Cannot generate schema for union of two types (excluding None): {annotation}"
                )
            non_none_types = [item for item in args if item is not type(None)]
            if len(non_none_types) != 1:
                raise ValueError(
                    f"Cannot generate schema for optional union with multiple concrete types: {annotation}"
                )
            return cls._parse_non_union_type(non_none_types[0]), True
        return cls._parse_non_union_type(annotation), False

    @classmethod
    def _parse_non_union_type(cls, annotation) -> _TypeSpec:
        """Parse non-union annotations into a normalized type spec.

        :param annotation: Annotation that is not a union.
        :returns: Parsed type spec.
        :raises ValueError: If type is unsupported or malformed.
        """

        origin = typing.get_origin(tp=annotation)
        if origin is list:
            args = typing.get_args(annotation)
            if len(args) != 1:
                raise ValueError(
                    f"Cannot generate schema for list with zero or multiple element types: {annotation}"
                )
            return _TypeSpec(kind="list", item_type=cls._parse_non_union_type(args[0]))
        if annotation in (str, float, int, bool):
            return _TypeSpec(kind="scalar", scalar_type=annotation)
        raise ValueError(f"Cannot generate schema for unsupported type: {annotation}")

    @classmethod
    def _field_specs(cls) -> list[_FieldSpec]:
        """Resolve dataclass fields into normalized field specs.

        :returns: Parsed field specifications in declaration order.
        """
        resolved_hints = typing.get_type_hints(cls, include_extras=True)
        field_specs = []
        for field in dataclasses.fields(cls):
            resolved_type = resolved_hints.get(field.name)

            if field.metadata.get("ignore_for_schema", False):
                continue

            type_spec, nullable = cls._parse_annotation(resolved_type)
            field_specs.append(
                _FieldSpec(name=field.name, type_spec=type_spec, nullable=nullable)
            )
        return field_specs

    @classmethod
    def _to_polars_type(cls, type_spec: _TypeSpec):
        """Convert internal type spec to Polars type.

        :param type_spec: Parsed type spec.
        :returns: Polars dtype for the spec.
        :raises ValueError: If list type has no element type.
        """

        scalar_map = {
            str: polars.String,
            float: polars.Float64,
            int: polars.Int64,
            bool: polars.Boolean,
        }
        if type_spec.kind == "scalar":
            return scalar_map[type_spec.scalar_type]
        if type_spec.item_type is None:
            raise ValueError("List type missing element type")
        return polars.List(cls._to_polars_type(type_spec.item_type))

    @classmethod
    def _to_pyarrow_type(cls, type_spec: _TypeSpec):
        """Convert internal type spec to PyArrow type.

        :param type_spec: Parsed type spec.
        :returns: PyArrow type for the spec.
        :raises ValueError: If list type has no element type.
        """

        scalar_map = {
            str: pyarrow.string(),
            float: pyarrow.float64(),
            int: pyarrow.int64(),
            bool: pyarrow.bool_(),
        }
        if type_spec.kind == "scalar":
            return scalar_map[type_spec.scalar_type]
        if type_spec.item_type is None:
            raise ValueError("List type missing element type")
        return pyarrow.list_(
            pyarrow.field(
                "item",
                cls._to_pyarrow_type(type_spec.item_type),
                nullable=False,
            )
        )

    @classmethod
    def _to_pyiceberg_type(
        cls, type_spec: _TypeSpec, id_allocator: _PyIcebergIdAllocator
    ):
        """Convert internal type spec to PyIceberg type.

        :param type_spec: Parsed type spec.
        :param id_allocator: Nested field ID allocator.
        :returns: PyIceberg type for the spec.
        :raises ValueError: If list type has no element type.
        """

        scalar_map = {
            str: pyiceberg.types.StringType(),
            float: pyiceberg.types.DoubleType(),
            int: pyiceberg.types.LongType(),
            bool: pyiceberg.types.BooleanType(),
        }
        if type_spec.kind == "scalar":
            return scalar_map[type_spec.scalar_type]
        if type_spec.item_type is None:
            raise ValueError("List type missing element type")
        return pyiceberg.types.ListType(
            element_id=id_allocator.next_nested_id(),
            element=cls._to_pyiceberg_type(type_spec.item_type, id_allocator),
        )

    @classmethod
    def as_polars_schema(cls) -> polars.Schema:
        """Build Polars schema from ``StructuredMetadata`` annotations.

        :returns: Polars schema matching dataclass field order.
        """

        field_specs = cls._field_specs()
        return polars.Schema(
            schema={
                field.name: cls._to_polars_type(field.type_spec)
                for field in field_specs
            }
        )

    @classmethod
    def as_pyarrow_schema(cls) -> pyarrow.Schema:
        """Build PyArrow schema from ``StructuredMetadata`` annotations.

        :returns: PyArrow schema with nullable flags preserved.
        """

        field_specs = cls._field_specs()
        return pyarrow.schema(
            fields=[
                pyarrow.field(
                    field.name,
                    type=cls._to_pyarrow_type(field.type_spec),
                    nullable=field.nullable,
                )
                for field in field_specs
            ]
        )

    @classmethod
    def as_pyiceberg_schema(cls) -> pyiceberg.schema.Schema:
        """Build PyIceberg schema from ``StructuredMetadata`` annotations.

        :returns: PyIceberg schema with deterministic field IDs.
        """

        field_specs = cls._field_specs()
        id_allocator = _PyIcebergIdAllocator(next_id=len(field_specs) + 1)
        return pyiceberg.schema.Schema(
            *[
                pyiceberg.types.NestedField(
                    field_id=index,
                    name=field.name,
                    field_type=cls._to_pyiceberg_type(field.type_spec, id_allocator),
                    required=not field.nullable,
                )
                for index, field in enumerate(field_specs, start=1)
            ]
        )

    @classmethod
    def to_arrow(
        cls,
        metadata: list[typing.Self],
    ) -> pyarrow.Table:
        """Convert a list of self to a validated pyarrow table"""
        return pyarrow.Table.from_pylist(
            [dataclasses.asdict(obj=metadata) for metadata in metadata],
            schema=cls.as_pyarrow_schema(),
        )
