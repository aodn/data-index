from unittest.mock import MagicMock

import polars as pl

# Assuming your class is imported like this:
import data_index.inventory_source


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

    delta_source = (
        data_index.inventory_source.DeltaIcebergTableInventorySource.model_construct(
            source=mock_source, sink=mock_sink
        )
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

    delta_source = (
        data_index.inventory_source.DeltaIcebergTableInventorySource.model_construct(
            source=mock_source,
            sink=mock_sink,
            left_on=["src_key"],
            right_on=["snk_key"],
        )
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

    delta_source = (
        data_index.inventory_source.DeltaIcebergTableInventorySource.model_construct(
            source=mock_source, sink=mock_sink
        )
    )

    assert delta_source.type == "delta_iceberg_table"
    assert delta_source.left_on == ("bucket", "key", "version_id")
    assert delta_source.right_on == ("bucket", "key", "version_id")
