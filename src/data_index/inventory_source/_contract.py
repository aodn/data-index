from __future__ import annotations

import polars

_REQUIRED_COLUMNS = ("bucket", "key", "version_id", "size")


def enforce_inventory_contract(inventory: polars.DataFrame) -> polars.DataFrame:
    """Enforce required inventory identity contract and normalize required dtypes."""
    missing = [
        column for column in _REQUIRED_COLUMNS if column not in inventory.columns
    ]
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")

    inventory = inventory.with_columns(
        polars.col("bucket").cast(polars.String),
        polars.col("key").cast(polars.String),
        polars.col("version_id").cast(polars.String),
        polars.col("size").cast(polars.Int64),
    )

    for column in ("bucket", "key", "version_id"):
        if inventory[column].null_count() > 0:
            raise ValueError(f"Inventory contains null `{column}` values")
        if len(inventory.filter(polars.col(column).str.strip_chars() == "")) > 0:
            raise ValueError(f"Inventory contains empty `{column}` values")

    if inventory["size"].null_count() > 0:
        raise ValueError("Inventory contains null `size` values")

    return inventory
