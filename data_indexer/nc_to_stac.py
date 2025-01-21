import logging

import cf_units
import numpy as np
import pandas as pd
import pystac
import s3fs
import xarray as xr
from pystac.extensions.datacube import Dimension, Variable

logger = logging.getLogger(__name__)


def json_type_conversion(obj):
    # Json can have trouble with numpy types, so we need to convert them
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, xr.DataArray):
        return obj.values.tolist()
    elif isinstance(obj, np.datetime64):
        return (
            str(obj.astype("datetime64[s]")) + "Z"
        )  # numpy datetimes are assumed to always be UTC
    elif isinstance(obj, dict):
        return {k: json_type_conversion(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_type_conversion(v) for v in obj]
    else:
        return obj


def classify_coord(coord: xr.DataArray) -> str:
    """
    Classify a coordinate as latitude, longitude, time, depth, or other.
    Parameters:
    coord (xarray.DataArray): The coordinate to classify.
    Returns:
    str: The classification of the coordinate.
    """
    attrs = coord.attrs
    standard_name = attrs.get('standard_name', '').lower()
    if standard_name in ['latitude', 'longitude', 'depth', 'time']:
        return standard_name

    unit = attrs.get('units')
    positive = attrs.get('positive', '').lower()
    if unit in [
        'degrees_north',
        'degree_north',
        'degree_N',
        'degrees_N',
        'degreeN',
        'degreesN',
    ]:
        return 'latitude'
    elif unit in [
        'degrees_east',
        'degree_east',
        'degree_E',
        'degrees_E',
        'degreeE',
        'degreesE',
    ]:
        return 'longitude'
    elif hasattr(coord, 'dt'):
        return 'time'
    else:
        try:
            unit = cf_units.Unit(unit)
        except ValueError:
            return 'other'
        if unit.is_time_reference():
            return 'time'
        elif (
            (
                unit.is_convertible('bar')
                or positive
                in [
                    'up',
                    'down',
                ]
            )
            and "air" not in standard_name
            and "stress" not in standard_name
        ):
            return 'depth'
        else:
            return 'other'


def nc_to_item(nc_file_path: str, collection: str, item_id: str = None) -> pystac.Item:
    """
    Converts a NetCDF file to a STAC Item.
    Parameters:
    nc_file_path (str): Path to the NetCDF file on s3.
    collection (str): The collection ID to which the item belongs.
    item_id (str, optional): The ID of the item. If None, it will be derived from the file name.
    Returns:
    pystac.Item: A STAC Item representing the NetCDF file.
    Raises:
    ValueError: If latitude, longitude, or time coordinates are not found in the dataset.
    Notes:
    - Accepts local file paths and S3 URIs.
    - Extracts spatial (latitude, longitude) and temporal (time) coordinates from the NetCDF file.
    - It creates a GeoJSON geometry and bounding box for the item.
    - The function sets various properties and assets for the STAC item.
    - It uses the DataCube extension to add dimensions and variables to the item.
    """

    logger.info(f'Processing {nc_file_path}')
    if item_id is None:
        item_id = nc_file_path.replace('/', '_').replace('.', '_')

    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(nc_file_path, 'rb') as f:
        ds = xr.open_dataset(f)

        # find latitude, longitude and time coordinates
        lat = []
        lon = []
        time = []
        depth = []
        for coord in ds.coords:
            coord_type = classify_coord(ds[coord])
            if coord_type == 'latitude':
                lat.append(coord)
            elif coord_type == 'longitude':
                lon.append(coord)
            elif coord_type == 'time':
                time.append(coord)
            elif coord_type == 'depth':
                depth.append(coord)
        if not lat or not lon or not time:
            # try looking at variables
            lat_vars = []
            lon_vars = []
            time_vars = []
            for var in ds.variables:
                coord_type = classify_coord(ds[var])
                if not lat and coord_type == 'latitude':
                    lat_vars.append(var)
                elif not lon and coord_type == 'longitude':
                    lon_vars.append(var)
                elif not time and coord_type == 'time':
                    time_vars.append(var)
            lat = lat or lat_vars
            lon = lon or lon_vars
            time = time or time_vars

        if not lat:
            raise ValueError('Latitude coordinate not found')
        if len(lat) > 1:
            logger.warning(f'Multiple latitude coordinates found: {lat}')
        lat = lat[0]

        if not lon:
            raise ValueError('Longitude coordinate not found')
        if len(lon) > 1:
            logger.warning(f'Multiple longitude coordinates found: {lon}')
        lon = lon[0]

        if not time:
            raise ValueError('Time coordinate not found')
        if len(time) > 1:
            logger.warning(f'Multiple time coordinates found: {time}')
        time = time[0]

        # Depth is optional
        if len(depth) > 1:
            logger.warning(f'Multiple depth coordinates found: {depth}')
        depth = depth[0] if depth else None

        # Create geoJSON box geometry
        bbox = [
            float(ds[lon].min()),
            float(ds[lat].min()),
            float(ds[lon].max()),
            float(ds[lat].max()),
        ]

        # TODO: work out 3D geometries

        if bbox[0] == bbox[2] and bbox[1] == bbox[3]:
            geometry = dict(type='Point', coordinates=[bbox[0], bbox[1]])
        else:
            geometry = dict(
                type='Polygon',
                coordinates=[
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ],
            )

        start_datetime = ds[time].min(skipna=True)
        if hasattr(start_datetime, 'dt'):
            start_datetime = pd.to_datetime(str(start_datetime.values), utc=True)
        else:
            start_datetime = pd.to_datetime(start_datetime, utc=True)
        end_datetime = ds[time].max(skipna=True)
        if hasattr(end_datetime, 'dt'):
            end_datetime = pd.to_datetime(str(end_datetime.values), utc=True)
        else:
            end_datetime = pd.to_datetime(end_datetime, utc=True)

        if start_datetime is None or end_datetime is None:
            raise ValueError('Invalid time coordinate')

        # We must provide either a single datetime or a start and end datetime
        if start_datetime == end_datetime:
            single_datetime = start_datetime
            start_datetime = end_datetime = None
        else:
            single_datetime = None

        properties = ds.attrs
        if 'abstract' in properties:
            # 'description' is the preferred name in STAC
            # For this we could use 'comment' from CF conventions or 'abstract' (from IMOS conventions?)
            properties['description'] = properties.pop(
                'abstract'
            )  # Should we duplicate it?
        # 'title' has the same name in CF and STAC

        # Other properties we could set:
        # - 'updated' (a datetime, do we need it?)
        # - 'deprecated' and related values: may be addressed through the version extension
        #   https://github.com/stac-extensions/version
        #   https://github.com/radiantearth/stac-spec/blob/master/best-practices.md#versioning-for-catalogs
        # - 'provider': should always be IMOS for now, could be useful later
        #   https://github.com/radiantearth/stac-spec/blob/v1.0.0/item-spec/common-metadata.md#provider-object
        # - 'instrument': designed for satellite data, but could be used for other sensors
        #   https://github.com/radiantearth/stac-spec/blob/v1.0.0/item-spec/common-metadata.md#instrument
        # - any other metadata that is not already covered by the STAC spec

        assets = (
            dict()
        )  # Dictionary of Asset objects, keys have no predefined meaning according to STAC
        assets['data'] = pystac.Asset(
            href=f's3://{nc_file_path}',  # We could use a different URI if we process the file locally
            media_type='application/netcdf',
            title='NetCDF data',
            # other standard roles are "thumbnail", "overview", "metadata"
            roles=['data'],
        )

        # Create base item without any extensions
        item = pystac.Item(
            id=item_id,
            geometry=geometry,
            bbox=bbox,
            datetime=single_datetime,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            properties=json_type_conversion(properties),
            assets=assets,
            collection=collection,
        )

        # DataCube extension for netcdf/zarr attributes and variables
        item.ext.add('cube')

        dimensions = dict()

        for dim in ds.dims:
            dim_info = dict()
            if dim == time:
                dim_info['type'] = 'temporal'
            elif dim == lon:
                dim_info['type'] = 'spatial'
                dim_info['axis'] = 'x'
            elif dim == lat:
                dim_info['type'] = 'spatial'
                dim_info['axis'] = 'y'
            elif dim == depth:
                dim_info['type'] = 'spatial'
                dim_info['axis'] = 'z'
            else:
                dim_info['type'] = 'other'

            # TODO: work out CRS information. Could use projection extension.
            # if dim_info['type'] == 'spatial':
            #     dim_info['reference_system'] = epsg

            dim_info['extent'] = [ds[dim].min().values, ds[dim].max().values]

            delta = ds[dim].diff(dim)
            if len(delta) > 1 and (delta[0] == delta[1:]).all():
                if dim_info['type'] == 'temporal':
                    dim_info['step'] = pd.to_timedelta(delta[0].values).isoformat()
                else:
                    dim_info['step'] = delta[0].values

            # Check for monotonic dimensions, not part of the indexing but could be easy to do here
            # increasing = (ds[dim][1:] >= ds[dim][:-1]).all()
            # decreasing = (ds[dim][1:] <= ds[dim][:-1]).all()
            # assert increasing or decreasing, f"Dimension '{dim}' is not monotonic"

            description = ds[dim].attrs.get('long_name') or ds[dim].attrs.get(
                'description'
            )
            if description:
                dim_info['description'] = description

            unit = ds[dim].attrs.get('units') or ds[dim].attrs.get('unit')
            if unit:
                dim_info['unit'] = unit

            dimensions[dim] = Dimension(json_type_conversion(dim_info))

        variables = dict()

        all_vars = list(ds.data_vars) + [c for c in ds.coords if c not in ds.dims]
        for var in all_vars:
            var_info = dict()
            var_info['dimensions'] = list(ds[var].dims)
            if var in ds.coords:
                var_info['type'] = 'auxiliary'
            else:
                var_info['type'] = 'data'

            description = ds[var].attrs.get('description') or ds[var].attrs.get(
                'long_name'
            )
            if description:
                var_info['description'] = description

            unit = ds[var].attrs.get('units') or ds[var].attrs.get('unit')
            if unit:
                var_info['unit'] = unit

            var_info['attrs'] = ds[var].attrs
            var_info['shape'] = list(ds[var].shape)

            variables[var] = Variable(json_type_conversion(var_info))

        # Add the dimensions and variables to the item using datacube extension
        item.ext.cube.apply(dimensions, variables)

        # Version extension for tracking changes to the item
        item.ext.add('version')
        item.ext.version.apply(version='1', deprecated=False)

    return item
