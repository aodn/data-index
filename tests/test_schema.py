import dataclasses
import re

import polars
import pyarrow
import pyiceberg.schema
import pyiceberg.types
import pytest

from data_index.schema.schema import (
    Schema,
    _PyIcebergIdAllocator,
    _TypeSpec,
)


@dataclasses.dataclass
class FlatScalarSchema(Schema):
    a_str: str
    an_int: int | None
    a_float: float
    a_bool: bool | None


@dataclasses.dataclass
class NestedListSchema(Schema):
    int_list: list[int]
    optional_str_list: list[str] | None


@dataclasses.dataclass
class NestedMapSchema(Schema):
    dimensions: dict[str, int] | None
    variables: list[str]


def test_as_polars_schema():
    """Verify Polars schema generation maps types accurately."""
    schema = FlatScalarSchema.as_polars_schema()

    assert isinstance(schema, polars.Schema)
    assert schema["a_str"] == polars.String
    assert schema["an_int"] == polars.Int64
    assert schema["a_float"] == polars.Float64
    assert schema["a_bool"] == polars.Boolean

    # Check nested list
    list_schema = NestedListSchema.as_polars_schema()
    assert list_schema["int_list"] == polars.List(polars.Int64)
    assert list_schema["optional_str_list"] == polars.List(polars.String)


def test_as_pyarrow_schema():
    """Verify PyArrow schema conversion and nullability configurations."""
    schema = FlatScalarSchema.as_pyarrow_schema()

    assert isinstance(schema, pyarrow.Schema)

    # Verify field types and nullability flags
    assert schema.field("a_str").type == pyarrow.string()
    assert not schema.field("a_str").nullable

    assert schema.field("an_int").type == pyarrow.int64()
    assert schema.field("an_int").nullable

    assert schema.field("a_float").type == pyarrow.float64()
    assert not schema.field("a_float").nullable

    assert schema.field("a_bool").type == pyarrow.bool_()
    assert schema.field("a_bool").nullable

    # Check nested list
    list_schema = NestedListSchema.as_pyarrow_schema()
    assert list_schema.field("int_list").type == pyarrow.list_(
        pyarrow.field("item", pyarrow.int64(), nullable=False)
    )


def test_as_pyiceberg_schema():
    """Verify Iceberg schema matches field IDs and requirements."""
    schema = FlatScalarSchema.as_pyiceberg_schema()

    assert isinstance(schema, pyiceberg.schema.Schema)

    # Iceberg increments root field positions sequentially (1 to 4)
    f1 = schema.find_field(1)
    assert f1.name == "a_str"
    assert f1.field_type == pyiceberg.types.StringType()
    assert f1.required

    f2 = schema.find_field(2)
    assert f2.name == "an_int"
    assert f2.field_type == pyiceberg.types.LongType()
    assert not f2.required

    # Check nested list ID allocation strategy
    list_schema = NestedListSchema.as_pyiceberg_schema()
    # Fields: int_list (id=1), optional_str_list (id=2)
    # Total fields = 2. Allocator starts next_id at 2 + 1 = 3.
    # Elements inside fields get nested IDs allocated from 3 onwards.
    int_list_field = list_schema.find_field(1)
    assert isinstance(int_list_field.field_type, pyiceberg.types.ListType)
    assert int_list_field.field_type.element_id == 3


def test_as_duckdb_schema():
    schema = FlatScalarSchema.as_duckdb_schema()
    assert schema == [
        ("a_str", "VARCHAR", False),
        ("an_int", "BIGINT", True),
        ("a_float", "DOUBLE", False),
        ("a_bool", "BOOLEAN", True),
    ]

    nested_schema = NestedMapSchema.as_duckdb_schema()
    assert nested_schema == [
        ("dimensions", "MAP(VARCHAR, BIGINT)", True),
        ("variables", "VARCHAR[]", False),
    ]


def test_invalid_union_types_raises_error():
    """Unions of multiple concrete types (excluding None) should fail."""

    @dataclasses.dataclass
    class BadUnionSchema(Schema):
        invalid_field: int | str

    with pytest.raises(
        ValueError, match="Cannot generate schema for union of two types"
    ):
        BadUnionSchema.as_polars_schema()


def test_multiple_concrete_types_with_optional_raises_error():
    """Optional unions with multiple concrete choices must fail."""

    @dataclasses.dataclass
    class BadOptionalSchema(Schema):
        invalid_field: int | str | None

    with pytest.raises(ValueError, match="Cannot generate schema for optional union"):
        BadOptionalSchema.as_polars_schema()


def test_unsupported_scalar_type_raises_error():
    """Complex or unsupported types like dict should throw an exception."""

    @dataclasses.dataclass
    class UnsupportedSchema(Schema):
        invalid_field: set

    with pytest.raises(ValueError, match="Cannot generate schema for unsupported type"):
        UnsupportedSchema.as_polars_schema()


def test_malformed_list_arguments_raises_error():
    """A list annotation with multiple inner types must fail."""

    @dataclasses.dataclass
    class BadListSchema(Schema):
        # This has an origin of 'list' but len(args) == 2
        invalid_list: list[int, str]

    with pytest.raises(
        ValueError,
        match="Cannot generate schema for list with zero or multiple element types",
    ):
        BadListSchema.as_polars_schema()


def test_naked_list_raises_unsupported_error():
    """A naked list without generic types is treated as an unsupported scalar."""

    @dataclasses.dataclass
    class NakedListSchema(Schema):
        invalid_list: list

    with pytest.raises(
        ValueError, match="Cannot generate schema for unsupported type: <class 'list'>"
    ):
        NakedListSchema.as_polars_schema()


def test_pyiceberg_id_allocator():
    """Directly validate sequential tracking behavior of the ID allocator."""
    allocator = _PyIcebergIdAllocator(next_id=10)
    assert allocator.next_nested_id() == 10
    assert allocator.next_nested_id() == 11
    assert allocator.next_id == 12


def test_converter_missing_item_type_raises_error():
    """Verify backend converters throw errors if an invalid list TypeSpec is provided."""
    malformed_spec = _TypeSpec(kind="list", item_type=None)
    allocator = _PyIcebergIdAllocator(next_id=1)

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Invalid or missing nested types for Polars spec: {malformed_spec}"
        ),
    ):
        Schema._to_polars_type(type_spec=malformed_spec)

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Invalid or missing nested types for PyArrow spec: {malformed_spec}"
        ),
    ):
        Schema._to_pyarrow_type(type_spec=malformed_spec)

    with pytest.raises(
        expected_exception=ValueError,
        match=re.escape(
            f"Invalid or missing nested types for Iceberg spec: {malformed_spec}"
        ),
    ):
        Schema._to_pyiceberg_type(type_spec=malformed_spec, id_allocator=allocator)


def test_as_duckdb_schema_requires_string_map_keys():
    @dataclasses.dataclass
    class InvalidMapKeySchema(Schema):
        attributes: dict[int, str]

    with pytest.raises(ValueError, match="DuckDB map keys must be strings"):
        InvalidMapKeySchema.as_duckdb_schema()
