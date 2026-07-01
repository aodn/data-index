import datetime
from unittest.mock import MagicMock

import freezegun
import polars as pl

import data_index.inventory_source.delta_iceberg_table


def test_delta_iceberg_table_inventory_anti_join():
    """
    Tests that the inventory method correctly performs an anti-join,
    returning rows from 'source' that are missing in 'sink'.
    """
    # Setup Mock Source Data (What we have)
    source_data = pl.DataFrame(
        {
            "bucket": ["b1", "b1", "b2"],
            "key": ["k1", "k2", "k3"],
            "version_id": ["v1", "v1", "v1"],
            "size": [100, 200, 300],
        }
    )

    # Setup Mock Sink Data (What is already processed/present)
    # k1 matches source, k9 is unique to sink (should be ignored)
    # k2 and k3 are missing here, so they should show up in the delta
    sink_data = pl.DataFrame(
        {
            "bucket": ["b1", "b4"],
            "key": ["k1", "k9"],
            "version_id": ["v1", "v1"],
            "size": [100, 999],
        }
    )

    mock_source = MagicMock()
    mock_source.inventory.return_value = source_data
    mock_sink = MagicMock()
    mock_sink.inventory.return_value = sink_data

    delta_source = data_index.inventory_source.delta_iceberg_table.DeltaIcebergTableInventorySource.model_construct(
        source=mock_source, sink=mock_sink
    )
    result_df = delta_source.inventory()

    # Expected: records for k2 and k3 from the source
    expected_df = pl.DataFrame(
        data={
            "bucket": ["b1", "b2"],
            "key": ["k2", "k3"],
            "version_id": ["v1", "v1"],
            "size": [200, 300],
        }
    )

    assert isinstance(result_df, pl.DataFrame)
    assert result_df.equals(other=expected_df)


def test_delta_iceberg_table_inventory_custom_join_keys():
    """
    Tests that custom 'left_on' and 'right_on' fields are respected
    during the anti-join operation.
    """

    # Source and sink have different column names representing the keys
    source_data = pl.DataFrame({"src_key": ["k1", "k2"]})
    sink_data = pl.DataFrame({"snk_key": ["k1"]})

    mock_source = MagicMock()
    mock_source.inventory.return_value = source_data
    mock_sink = MagicMock()
    mock_sink.inventory.return_value = sink_data

    delta_source = data_index.inventory_source.delta_iceberg_table.DeltaIcebergTableInventorySource.model_construct(
        source=mock_source,
        sink=mock_sink,
        left_on=["src_key"],
        right_on=["snk_key"],
    )

    result_df = delta_source.inventory()
    expected_df = pl.DataFrame(data={"src_key": ["k2"]})

    assert result_df.equals(other=expected_df)


def test_delta_iceberg_table_pydantic_defaults():
    """
    Tests that Pydantic defaults for 'type', 'left_on', and 'right_on'
    are correctly assigned if omitted.
    """
    mock_source = MagicMock()
    mock_sink = MagicMock()

    delta_source = data_index.inventory_source.delta_iceberg_table.DeltaIcebergTableInventorySource.model_construct(
        source=mock_source, sink=mock_sink
    )

    assert delta_source.type == "delta_iceberg_table"
    assert delta_source.left_on == ["bucket", "key", "version_id"]
    assert delta_source.right_on == ["bucket", "key", "version_id"]


class MockDatetime:
    # This makes MockDatetime.datetime return MockDatetime itself,
    # satisfying the source code's `datetime.datetime.now()` syntax chain.
    @property
    def datetime(self):
        return self

    def now(self, tz=None):
        return datetime.datetime(2026, 7, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)

    @property
    def timezone(self):
        return datetime.timezone


# 2. Patch the target module's reference to datetime.datetime
@freezegun.freeze_time(time_to_freeze="2026-07-01T12:00:00")
def test_lookback_config_time_generation():
    config = data_index.inventory_source.delta_iceberg_table.LookbackConfig(
        days=1, hours=2, column_name="last_modified"
    )

    expected_iso = "2026-06-30T10:00:00+00:00"

    assert config.lookback_timestamp == expected_iso
    assert config.lookback_filter == "last_modified >= '2026-06-30T10:00:00+00:00'"


@freezegun.freeze_time("2026-07-01T12:00:00+00:00")
def test_delta_iceberg_inventory_appends_lookback_filters():
    # Arrange Mock DataFrames and Table Sources
    mock_source_df = pl.DataFrame(
        {"bucket": ["b1"], "key": ["k1"], "version_id": ["v1"]},
        schema={"bucket": pl.String, "key": pl.String, "version_id": pl.String},
    )

    # Explicitly enforce schema
    mock_sink_df = pl.DataFrame(
        {"bucket": [], "key": [], "version_id": []},
        schema={"bucket": pl.String, "key": pl.String, "version_id": pl.String},
    )

    # Create Mock Iceberg Sources
    mock_source = MagicMock()
    mock_source.inventory.return_value = mock_source_df
    # Initialize table_scan_config with no row_filter initially
    mock_source.table_scan_config = MagicMock()
    mock_source.table_scan_config.row_filter = None

    mock_sink = MagicMock()
    mock_sink.inventory.return_value = mock_sink_df

    # Configure a 1 day, 2 hour lookback config -> target time: 2026-06-30T10:00:00+00:00
    lookback_config = data_index.inventory_source.delta_iceberg_table.LookbackConfig(
        days=1, hours=2, column_name="last_modified"
    )
    expected_lookback_filter = "last_modified >= '2026-06-30T10:00:00+00:00'"

    # Case A: Verify it writes directly when there is NO existing filter
    inventory_source_no_filter = data_index.inventory_source.delta_iceberg_table.DeltaIcebergTableInventorySource.model_construct(
        source=mock_source,
        sink=mock_sink,
        lookback_config=lookback_config,
    )

    # Trigger the inventory call
    inventory_source_no_filter.inventory()

    assert mock_source.table_scan_config.row_filter == expected_lookback_filter

    # Case B: Verify it appends using AND when there IS an existing filter
    mock_source.table_scan_config.row_filter = "status = 'active'"

    inventory_source_with_filter = data_index.inventory_source.delta_iceberg_table.DeltaIcebergTableInventorySource.model_construct(
        source=mock_source,
        sink=mock_sink,
        lookback_config=lookback_config,
    )

    # Trigger the inventory call again
    result_df = inventory_source_with_filter.inventory()

    expected_combined_filter = f"(status = 'active') AND {expected_lookback_filter}"
    assert mock_source.table_scan_config.row_filter == expected_combined_filter

    # Verify the anti-join executed correctly (mock_source_df elements remain)
    assert result_df.shape == (1, 3)
    assert result_df.equals(mock_source_df)
