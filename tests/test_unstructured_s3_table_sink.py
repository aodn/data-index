import json

import pytest

from data_index._collection import derive_facility
from data_index.iceberg_config import IcebergTableConfig, SqliteCatalogConfig
from data_index.protocols import ObjectReference
from data_index.schema.metadata import UnstructuredMetadata
from data_index.unstructured_sink.s3_table_sink import UnstructuredS3TableSink

NAMESPACE = "test_ns"
TABLE_NAME = "unstructured_metadata"


@pytest.fixture
def table_config(tmp_path):
    catalog_config = SqliteCatalogConfig(
        uri=f"sqlite:///{tmp_path}/catalog.db",
        warehouse=str(tmp_path / "warehouse"),
    )
    config = IcebergTableConfig(
        catalog_config=catalog_config,
        namespace=NAMESPACE,
        table_name=TABLE_NAME,
    )
    UnstructuredS3TableSink(iceberg_table_config=config).provision()
    return config


def test_provision_is_idempotent(table_config):
    UnstructuredS3TableSink(
        iceberg_table_config=table_config
    ).provision()  # table already exists — must not raise


def test_provision_sets_schema_version_table_property(table_config):
    table = table_config.load()

    assert table.properties["schema_version"] == "1"


def test_writes_identity_facility_and_json_metadata(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    object_references = [
        ObjectReference(
            bucket="test",
            key="IMOS/ANMN/a.nc",
            version_id="0",
            size=0,
            xarray_handle=None,
            extraction_result=None,
        ),
        ObjectReference(
            bucket="test",
            key="IMOS/ANMN/b.nc",
            version_id="0",
            size=0,
            xarray_handle=None,
            extraction_result=None,
        ),
    ]

    unstructured_metadata = [
        UnstructuredMetadata(
            bucket=object_reference.bucket,
            key=object_reference.key,
            version_id=object_reference.version_id,
            hash=object_reference.hash,
            facility=derive_facility(object_reference.key),
            metadata="{}",
        )
        for object_reference in object_references
    ]

    sink.write(unstructured_metadata)

    df = table_config.load().scan().to_pandas()
    assert len(df) == 2
    assert set(df["bucket"]) == {"test"}
    assert set(df["key"]) == {"IMOS/ANMN/a.nc", "IMOS/ANMN/b.nc"}
    assert set(df["version_id"]) == {"0", "0"}
    assert set(df["facility"]) == {"ANMN"}


def test_upserts_on_subsequent_writes(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    # Define a shared ObjectReference
    obj_ref = ObjectReference(
        bucket="imos-data",
        key="IMOS/ANMN/a.nc",
        version_id="v1",
        size=0,
        xarray_handle=None,
        extraction_result=None,
    )

    # First write (Initial insertion)
    meta_v1 = UnstructuredMetadata(
        bucket=obj_ref.bucket,
        key=obj_ref.key,
        version_id=obj_ref.version_id,
        hash="initial-hash",
        facility=derive_facility(obj_ref.key),
        metadata=json.dumps({"x": 1}),
    )
    sink.write([meta_v1])

    # Second write (Upserting with updated metadata)
    meta_v2 = UnstructuredMetadata(
        bucket=obj_ref.bucket,
        key=obj_ref.key,
        version_id=obj_ref.version_id,
        hash="initial-hash",
        facility=derive_facility(obj_ref.key),
        metadata=json.dumps({"x": 2}),
    )
    sink.write([meta_v2])

    # Assertions to ensure deduplication/upsert logic worked
    df = table_config.load().scan().to_pandas()
    assert len(df) == 1
    assert json.loads(df["metadata"].iloc[0]) == {"x": 2}


def test_upsert_replaces_existing_and_inserts_new_rows(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    # Define ObjectReferences
    obj_ref_a = ObjectReference(
        bucket="imos-data",
        key="IMOS/ANMN/a.nc",
        version_id="v1",
        size=0,
        xarray_handle=None,
        extraction_result=None,
    )
    obj_ref_b = ObjectReference(
        bucket="imos-data",
        key="IMOS/ANMN/b.nc",
        version_id="v2",
        size=0,
        xarray_handle=None,
        extraction_result=None,
    )

    # First write (Initial insertion of row A)
    meta_a_v1 = UnstructuredMetadata(
        bucket=obj_ref_a.bucket,
        key=obj_ref_a.key,
        version_id=obj_ref_a.version_id,
        hash="hash-a",
        facility=derive_facility(obj_ref_a.key),
        metadata=json.dumps({"x": 1}),
    )
    sink.write([meta_a_v1])

    # Second write (Upsert row A and insert new row B)
    meta_a_v2 = UnstructuredMetadata(
        bucket=obj_ref_a.bucket,
        key=obj_ref_a.key,
        version_id=obj_ref_a.version_id,
        hash="hash-a",
        facility=derive_facility(obj_ref_a.key),
        metadata=json.dumps({"x": 2}),
    )
    meta_b_v1 = UnstructuredMetadata(
        bucket=obj_ref_b.bucket,
        key=obj_ref_b.key,
        version_id=obj_ref_b.version_id,
        hash="hash-b",
        facility=derive_facility(obj_ref_b.key),
        metadata=json.dumps({"x": 3}),
    )
    sink.write([meta_a_v2, meta_b_v1])

    # Assertions
    df = table_config.load().scan().to_pandas()
    metadata_by_identity = {
        (row["bucket"], row["key"], row["version_id"]): json.loads(row["metadata"])
        for _, row in df.iterrows()
    }

    assert len(df) == 2
    assert metadata_by_identity[("imos-data", "IMOS/ANMN/a.nc", "v1")] == {"x": 2}
    assert metadata_by_identity[("imos-data", "IMOS/ANMN/b.nc", "v2")] == {"x": 3}


def test_empty_write_is_noop(table_config):
    sink = UnstructuredS3TableSink(iceberg_table_config=table_config)

    sink.write({})

    df = table_config.load().scan().to_pandas()
    assert len(df) == 0
