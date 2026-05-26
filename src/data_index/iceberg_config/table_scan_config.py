import pydantic


class IcebergTableScanConfig(pydantic.BaseModel):
    """Configuration for an Iceberg table scan."""

    row_filter: str | None = pydantic.Field(
        default=None, description="Filter expression for rows to include in the scan."
    )
    selected_fields: tuple[str, ...] = pydantic.Field(
        default=(
            "bucket",
            "key",
            "version_id",
            "size",
            "sequence_number",
            "is_delete_marker",
        ),
        description="List of columns to project.",
    )
    case_sensitive: bool = pydantic.Field(
        default=True,
        description="Whether column name lookups should be case sensitive.",
    )
    snapshot_id: int | None = pydantic.Field(
        default=None,
        description="The ID of the snapshot to read for time-travel queries.",
    )
    limit: int | None = pydantic.Field(
        default=None, description="Maximum number of rows to return."
    )
