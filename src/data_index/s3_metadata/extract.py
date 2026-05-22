import pathlib

import polars
import prefect
import pyarrow.parquet
import pyiceberg.catalog
import pyiceberg.table
import pyiceberg.expressions
import pydantic

INVENTORY_TABLE_SCHEMA = pyarrow.schema(
    [
        pyarrow.field(
            "bucket",
            pyarrow.large_string(),
            nullable=False,
            metadata={"doc": "The general purpose bucket name."},
        ),
        pyarrow.field(
            "key",
            pyarrow.large_string(),
            nullable=False,
            metadata={
                "doc": "The object key name (or key) that uniquely identifies the object in the bucket."
            },
        ),
        pyarrow.field(
            "sequence_number",
            pyarrow.large_string(),
            nullable=False,
            metadata={
                "doc": "The sequence number, which is an ordinal that's included in the records for a given object. To order records of the same bucket and key, you can sort on sequence_number."
            },
        ),
        pyarrow.field(
            "version_id",
            pyarrow.large_string(),
            nullable=True,
            metadata={
                "doc": "The object's version ID. Amazon S3 assigns a version number to objects added to the bucket."
            },
        ),
        pyarrow.field(
            "is_delete_marker",
            pyarrow.bool_(),
            nullable=True,
            metadata={
                "doc": "The object's delete marker status. True if the object is a delete marker."
            },
        ),
        pyarrow.field(
            "size",
            pyarrow.int64(),
            nullable=True,
            metadata={
                "doc": "The object size in bytes. If is_delete_marker is True, the size is 0."
            },
        ),
        pyarrow.field(
            "last_modified_date",
            pyarrow.timestamp("us"),
            nullable=True,
            metadata={
                "doc": "The object creation date or the last modified date, whichever is the latest."
            },
        ),
        pyarrow.field(
            "e_tag",
            pyarrow.large_string(),
            nullable=True,
            metadata={
                "doc": "The entity tag (ETag), which is a hash of the object contents."
            },
        ),
        pyarrow.field(
            "storage_class",
            pyarrow.large_string(),
            nullable=True,
            metadata={"doc": "The storage class that's used for storing the object."},
        ),
        pyarrow.field(
            "is_multipart",
            pyarrow.bool_(),
            nullable=True,
            metadata={"doc": "True if the object was uploaded as a multipart upload."},
        ),
        pyarrow.field(
            "encryption_status",
            pyarrow.large_string(),
            nullable=True,
            metadata={"doc": "The object's server-side encryption status."},
        ),
        pyarrow.field(
            "is_bucket_key_enabled",
            pyarrow.bool_(),
            nullable=True,
            metadata={"doc": "The object's S3 Bucket Key enablement status."},
        ),
        pyarrow.field(
            "kms_key_arn",
            pyarrow.large_string(),
            nullable=True,
            metadata={"doc": "The ARN for the KMS key used for encryption."},
        ),
        pyarrow.field(
            "checksum_algorithm",
            pyarrow.large_string(),
            nullable=True,
            metadata={
                "doc": "The algorithm used to create the checksum for the object."
            },
        ),
        pyarrow.field(
            "object_tags",
            pyarrow.map_(
                pyarrow.large_string(),
                pyarrow.large_string(),
            ),
            nullable=True,
            metadata={
                "doc": "Object tags associated with the object as key-value pairs."
            },
        ),
        pyarrow.field(
            "user_metadata",
            pyarrow.map_(
                pyarrow.large_string(),
                pyarrow.large_string(),
            ),
            nullable=True,
            metadata={
                "doc": "User metadata associated with the object as key-value pairs."
            },
        ),
    ]
)


class S3TablesConfig(pydantic.BaseModel):
    region: str
    arn: str = pydantic.Field(
        pattern=r"arn:aws[-a-z0-9]*:[a-z0-9]+:[-a-z0-9]*:[0-9]{12}:bucket/[a-z0-9_-]{3,63}"
    )
    namespace: str
    table_name: str

    @property
    def uri(self) -> str:
        return f"https://s3tables.{self.region}.amazonaws.com/iceberg"


class TableScanConfig(pydantic.BaseModel):
    """
    Configuration for an Iceberg table scan using Pydantic.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    row_filter: str | pyiceberg.expressions.BooleanExpression | None = pydantic.Field(
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


@prefect.task
def load_table(
    s3_table_config: S3TablesConfig,
) -> pyiceberg.table.Table:

    # Load the catalog
    catalog = pyiceberg.catalog.load_catalog(
        # catalog = load_catalog(
        "s3tables_catalog",
        **{
            "type": "rest",
            "uri": s3_table_config.uri,
            "warehouse": s3_table_config.arn,
            "rest.sigv4-enabled": "true",
            "rest.signing-name": "s3tables",
            "rest.signing-region": s3_table_config.region,
        },
    )

    table = catalog.load_table(("b_imos-data", s3_table_config.table_name))

    # Load the table
    return table


@prefect.task
def sink_table(
    table: pyiceberg.table.Table,
    table_scan_config: TableScanConfig,
    inventory_parquet_path: pathlib.Path,
):
    logger = prefect.get_run_logger()

    # Pre-construct the parent directory to save batches in
    inventory_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Table scan config:\n{table_scan_config.model_dump_json(indent=4)}")

    # Set up the batch reader
    with table.scan(
        **table_scan_config.model_dump(exclude_none=True)
    ).to_arrow_batch_reader() as batches:
        # Set up the batch writer
        with pyarrow.parquet.ParquetWriter(
            where=inventory_parquet_path,
            schema=pyarrow.schema(
                [
                    field
                    for field in INVENTORY_TABLE_SCHEMA
                    if field.name in table_scan_config.selected_fields
                ]
            ),
            compression="zstd",
        ) as writer:
            # Write batches
            for batch in batches:
                writer.write_batch(batch=batch)
                logger.info(
                    f"Wrote batch of in memory size {batch.get_total_buffer_size()} bytes..."
                )


@prefect.task
def extract(
    s3_table_config: S3TablesConfig = S3TablesConfig(
        region="ap-southeast-2",
        arn="arn:aws:s3tables:ap-southeast-2:104044260116:bucket/aws-s3",
        namespace="b_imos-data",
        table_name="inventory",
    ),
    table_scan_config: TableScanConfig = TableScanConfig(),
    inventory_parquet_path: pathlib.Path = pathlib.Path("imos-data.inventory.parquet"),
) -> polars.LazyFrame:

    table = load_table(s3_table_config)
    sink_table(
        table=table,
        table_scan_config=table_scan_config,
        inventory_parquet_path=inventory_parquet_path,
    )

    return polars.scan_parquet(inventory_parquet_path)


if __name__ == "__main__":
    extract()
