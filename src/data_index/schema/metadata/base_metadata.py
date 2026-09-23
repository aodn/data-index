import dataclasses
import decimal
import json
import math
import typing

import data_index.schema
from data_index.schema.schema import (
    DynamoDBAttributeType,
    DynamoDBTypeSpec,
)


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
        dynamodb_attribute_type: DynamoDBAttributeType,
    ) -> typing.Any:
        """Normalize scalar values by DynamoDB scalar family.

        - ``S`` and ``BOOL`` pass through unchanged.
        - ``N`` must not receive Python ``float``; boto3 expects ``Decimal``.
        """
        if dynamodb_attribute_type in ("S", "BOOL"):
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
        value: list,
    ) -> str:
        """Serialize a DynamoDB ``L`` value to compact JSON text."""
        if not isinstance(value, list):
            raise TypeError(f"Expected list for DynamoDB type 'L', got {type(value)}")
        # Compact separators reduce item size; sorted keys keep output deterministic.
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _coerce_map_value(
        cls,
        value: dict,
    ) -> str:
        """Serialize a DynamoDB ``M`` value to compact JSON text."""
        if not isinstance(value, dict):
            raise TypeError(f"Expected dict for DynamoDB type 'M', got {type(value)}")
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Expected string map keys for DynamoDB type 'M'")
        # Compact separators reduce item size; sorted keys keep output deterministic.
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @classmethod
    def _coerce_dynamodb_value(
        cls,
        value: typing.Any,
        type_spec: DynamoDBTypeSpec,
    ) -> typing.Any:
        """Normalize one value according to a DynamoDB type contract.

        - Dispatches scalar conversion for ``S``/``N``/``BOOL``.
        - Serializes ``L``/``M`` values to JSON text for string storage.
        - Keeps runtime conversion policy decoupled from Schema parsing.
        """
        if value is None:
            return None

        dynamodb_type = type_spec.dynamodb_type
        match dynamodb_type:
            case "N" | "S" | "BOOL":
                return cls._coerce_scalar(value, dynamodb_attribute_type=dynamodb_type)
            case "L":
                return cls._coerce_list_value(value)
            case "M":
                return cls._coerce_map_value(value)
            case _:
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
        contract and applies runtime normalization (float -> Decimal, list/map -> JSON text).
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
