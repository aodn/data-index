import dataclasses
import decimal
import math
import typing

import data_index.schema
from data_index.schema.schema import DynamoDBTypeSpec


@dataclasses.dataclass(
    kw_only=True,
    frozen=True,
)
class BaseMetadata(data_index.schema.Schema):
    """Metadata row schema to inherit from"""

    bucket: str
    key: str
    version_id: str
    hash: str
    file_format: str
    facility: str

    @classmethod
    def _coerce_scalar(
        cls,
        value: typing.Any,
        *,
        dynamodb_type: str,
    ) -> typing.Any:
        """Normalize scalar values by DynamoDB scalar family.

        - ``S`` and ``BOOL`` pass through unchanged.
        - ``N`` must not receive Python ``float``; boto3 expects ``Decimal``.
        """
        if dynamodb_type in ("S", "BOOL"):
            return value
        if isinstance(value, float):
            # DynamoDB numbers cannot represent NaN/Infinity, so we normalize to
            # ``None`` and let include_nulls policy decide whether to retain it.
            if not math.isfinite(value):
                return None
            return decimal.Decimal(str(value))
        return value

    @classmethod
    def _coerce_list_value(
        cls,
        value: typing.Any,
        *,
        type_spec: DynamoDBTypeSpec,
        include_nulls: bool,
    ) -> list[typing.Any]:
        """Normalize a DynamoDB ``L`` value recursively.

        - Isolates list-specific validation and recursion.
        - Keeps null-filtering policy (`include_nulls`) explicit for nested data.
        """
        if not isinstance(value, list):
            raise ValueError(
                f"Expected list for DynamoDB type 'L', got {type(value)}"
            )
        if type_spec.item_type is None:
            raise ValueError("List DynamoDB type spec missing item_type")

        converted_items = [
            cls._coerce_dynamodb_value(
                item,
                type_spec.item_type,
                include_nulls=include_nulls,
            )
            for item in value
        ]
        if include_nulls:
            return converted_items
        return [item for item in converted_items if item is not None]

    @classmethod
    def _coerce_map_value(
        cls,
        value: typing.Any,
        *,
        type_spec: DynamoDBTypeSpec,
        include_nulls: bool,
    ) -> dict[str, typing.Any]:
        """Normalize a DynamoDB ``M`` value recursively.

        - Enforces DynamoDB string-keyed map constraints.
        - Applies recursive normalization to map values using the spec contract.
        """
        if not isinstance(value, dict):
            raise ValueError(
                f"Expected dict for DynamoDB type 'M', got {type(value)}"
            )
        if type_spec.value_type is None:
            raise ValueError("Map DynamoDB type spec missing value_type")

        converted_map: dict[str, typing.Any] = {}
        for key, map_value in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Expected string map key for DynamoDB type 'M', got {type(key)}"
                )
            converted_value = cls._coerce_dynamodb_value(
                map_value,
                type_spec.value_type,
                include_nulls=include_nulls,
            )
            if converted_value is None and not include_nulls:
                continue
            converted_map[key] = converted_value
        return converted_map

    @classmethod
    def _coerce_dynamodb_value(
        cls,
        value: typing.Any,
        type_spec: DynamoDBTypeSpec,
        *,
        include_nulls: bool,
    ) -> typing.Any:
        """Normalize one value according to a DynamoDB type contract.

        - Central recursive dispatcher for ``S``/``N``/``BOOL``/``L``/``M``.
        - Keeps runtime conversion policy decoupled from Schema parsing.
        """
        if value is None:
            return None

        dynamodb_type = type_spec.dynamodb_type
        if dynamodb_type in ("N", "S", "BOOL"):
            return cls._coerce_scalar(value, dynamodb_type=dynamodb_type)
        if dynamodb_type == "L":
            return cls._coerce_list_value(
                value,
                type_spec=type_spec,
                include_nulls=include_nulls,
            )
        if dynamodb_type == "M":
            return cls._coerce_map_value(
                value,
                type_spec=type_spec,
                include_nulls=include_nulls,
            )
        raise ValueError(f"Unsupported DynamoDB type spec: {dynamodb_type}")

    @classmethod
    def _get_dynamodb_row_item(
        cls,
        source: dict,
        dynamodb_type_spec: DynamoDBTypeSpec,
        include_nulls: bool,
    ):

        # Set up new container
        row_item: dict[str, typing.Any] = {}

        # Coerce all the values
        for field_name, type_spec in dynamodb_type_spec.items():
            converted_value = cls._coerce_dynamodb_value(
                source[field_name],
                type_spec,
                include_nulls=include_nulls,
            )

            # If `include_nulls` is false and the coerced value is `None`
            # we don't add it to the dict
            if converted_value is None and not include_nulls:
                continue
            row_item[field_name] = converted_value

        return row_item

    def as_dynamodb_item(
        self,
        include_nulls: bool = False,
    ) -> dict[str, typing.Any]:
        """Return a DynamoDB-compatible item from this metadata row.

        Conversion uses ``Schema.as_dynamodb_type_spec()`` as the structural
        contract and applies runtime numeric normalization (float -> Decimal).
        """
        # We intentionally read from __dict__ (instead of dataclasses.asdict)
        # to avoid deep-copy overhead on hot per-row serialization paths.
        source = self.__dict__
        dynamodb_type_spec = self.as_dynamodb_type_spec()
        row_item = self._get_dynamodb_row_item(
            source=source,
            dynamodb_type_spec=dynamodb_type_spec,
            include_nulls=include_nulls,
        )

        return row_item