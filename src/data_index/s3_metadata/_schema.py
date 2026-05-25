import pyarrow

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
