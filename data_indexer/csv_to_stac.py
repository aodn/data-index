import logging

import pandas as pd
import pystac
from pystac.extensions.table import Column

logger = logging.getLogger(__name__)


def csv_to_item(
    csv_file_path: str, collection: str, item_id: str = None
) -> pystac.Item:
    """
    Converts a CSV file to a STAC item.
    Parameters:
    csv_file_path (str): The path to the CSV file on s3.
    collection (str): The collection ID to which the item belongs.
    item_id (str, optional): The ID of the item. If None, it will be derived from the file name.
    Returns:
    pystac.Item: A STAC Item representing the NetCDF file.
    Raises:
    ValueError: If latitude, longitude, or time coordinates are not found in the data.
    Notes:
    - Accepts local file paths and S3 URIs.
    - Needs to load the entire CSV file into memory.
    - Extracts spatial (latitude, longitude) and temporal (time) coordinates from the file, based on column names.
    - It creates a GeoJSON geometry and bounding box for the item.
    - The function sets various properties and assets for the STAC item.
    - It uses the Table extension to list columns and their types.
    """

    logger.info(f'Processing CSV file {csv_file_path}')
    if item_id is None:
        item_id = csv_file_path.replace('/', '_').replace('.', '_')
    df = pd.read_csv(csv_file_path)

    lat_names = ['latitude', 'lat']
    lon_names = ['longitude', 'lon']
    time_names = ['time', 'date']

    lat = None
    lon = None
    time = None

    for column in df.columns:
        if column.lower() in lat_names:
            lat = column
        elif column.lower() in lon_names:
            lon = column
        elif column.lower() in time_names:
            time = column
        elif time is None:
            for time_name in time_names:
                if time_name in column.lower():
                    # This can produce wrong results if there are multiple
                    # columns with time or date in the name, but it's a simple
                    # heuristic that works in most cases.
                    time = column
                    break

    if time is None:
        raise ValueError(f"Time column not found in the CSV file {csv_file_path}.")

    if lat is None or lon is None:
        logger.warning(
            f"Latitude or longitude columns not found in the CSV file {csv_file_path}."
        )
        bbox = None
        geometry = None
    else:
        # Compute bounding box and geometry
        bbox = [df[lon].min(), df[lat].min(), df[lon].max(), df[lat].max()]
        geometry = {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]
            ],
        }

    # Compute time range
    df[time] = pd.to_datetime(df[time])
    start_datetime = df[time].min()
    end_datetime = df[time].max()
    single_datetime = None
    if start_datetime == end_datetime:
        single_datetime = start_datetime
        start_datetime = None
        end_datetime = None

    # Create STAC item
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=single_datetime,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        properties={},
        collection=collection,
    )

    # Add table extension
    item.ext.add("table")
    columns = []

    for column in df.columns:
        columns.append(Column(dict(name=column, col_type=str(df[column].dtype))))
        # We could add min/max values for each column here

    item.ext.table.columns = columns
    item.ext.table.row_count = len(df)

    # Add assets
    item.add_asset(
        "data",
        pystac.Asset(
            href=csv_file_path,
            media_type="text/csv",
            roles=["data"],
            title="CSV data",
        ),
    )

    return item
