import logging
import re
from typing import List

import boto3
import numpy as np
import pandas as pd
import pystac
import requests
import s3fs
import xarray as xr
from pystac.extensions.datacube import Dimension, Variable

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
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
        item_id = nc_file_path.split('/')[-1].split('.')[0]

    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(nc_file_path, 'rb') as f:
        ds = xr.open_dataset(f)

        # find latitude, longitude and time coordinates
        lat = None
        lon = None
        time = None
        depth = None
        for coord in ds.coords:
            if ds[coord].attrs.get('standard_name') == 'latitude':
                lat = coord
            if ds[coord].attrs.get('standard_name') == 'longitude':
                lon = coord
            if ds[coord].attrs.get('standard_name') == 'time':
                time = coord
            if ds[coord].attrs.get('standard_name') == 'depth':
                depth = coord
        if lat is None or lon is None or time is None:
            raise ValueError('Could not find spatiotemporal coordinates')

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

        # We must provide either a single datetime or a start and end datetime
        start_datetime = pd.to_datetime(ds[time].min().values, utc=True)
        end_datetime = pd.to_datetime(ds[time].max().values, utc=True)
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

        properties['created'] = ds.attrs.pop('date_created')  # Same question here
        # This should more properly be the creation date of this STAC item, while the
        # creation date of the data file should be stored in the asset. If we plan to
        # keep the items in sync with the data files, setting it here can make it easier
        # for searching.

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

            description = ds[dim].attrs.get('long_name') or ds[dim].attrs.get('description')
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

            description = ds[var].attrs.get('description') or ds[var].attrs.get('long_name')
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


def get_collection_id(s3_uri: str) -> str:
    """
    Get the collection ID for an S3 object.
    Args:
        s3_uri (str): An S3 URI.
    Returns:
        str: The collection ID.
    """
    regex_to_collection_id = {
        "^IMOS/ANMN/.*/Wave/.*_FV01_.*(ADCP|AWAC).*\\.nc": "aaebf991-b79d-4670-a1c5-a0de9bf649ce",
        "^IMOS/ANMN/.*/Temperature/.*_FV01_.*\\.nc$": "7e13b5f3-4a70-4e31-9e95-335efa491c5c",
        "^IMOS/ANMN/.*/CTD_timeseries/.*_FV01_.*(SBE|XR-420).*_END-.*\\.nc$": "7e13b5f3-4a70-4e31-9e95-335efa491c5c",
        "^IMOS/ANMN/.*/Wave/.*_FV00_.*realtime.*\\.nc": "72dbe843-2fb1-4b2e-8b7e-4661d857affb",
        "^IMOS/ANMN/.*/Meteorology/.*_FV00_.*realtime.*\\.nc": "f3910f5c-c568-4af0-b773-13c0e57ada6b",
        "^IMOS/ANMN/.*/Biogeochem_timeseries/.*_FV00_.*realtime.*\\.nc": "13b3900f-2623-463c-98a7-ea60ac8e61ae",
        "^IMOS/ANMN/NRS/.*/aggregated_products/IMOS_ANMN-NRS_.*_FV02_.*\\.nc": "d3feff71-ebe1-4b66-91c6-149beceef205",
        "^IMOS/ANMN/NRS/REAL_TIME/.*_channel_.*/IMOS_ANMN_.*_FV0.*": "006bb7dc-860b-4b89-bf4c-6bd930bd35b7",
        "^AIMS/Marine_Monitoring_Program/CTD_profiles/.*/.*_FV01_.*\\.nc": "acad78d1-e235-45e6-8f27-0a00184e2ca9",
        "^IMOS/ANMN/.*/Biogeochem_profiles/.*_FV01_.*\\.nc": "7b901002-b1dc-46c3-89f2-b4951cedca48",
        "^NSW-OEH/Manly_Hydraulics_Laboratory/Wave/.*": "bb7e9d82-3b9c-44c6-8e93-1ee9fd30bf21",
        "^IMOS/ANMN/.*_FV02_.*-burst-averaged.*\\.nc$": "8964658c-6ee1-4015-9bae-2937dfcc6ab9",
        "^IMOS/ANMN/AM/.*-delayed_.*\\.nc$": "89b495cc-7382-43c0-abef-d1e66738a924",
        "^IMOS/ANMN/AM/.*-realtime\\.nc$": "4d3d4aca-472e-4616-88a5-df0f5ab401ba",
        "^IMOS/ANMN/.*/Velocity/.*FV01.*\\.nc$": "ae86e2f5-eaaf-459e-a405-e654d85adb9c",
        "^IMOS/ANMN/Acoustic/metadata/update.*\\.csv$": "e850651b-d65d-495b-8182-5dde35919616",
    }
    for regex, collection_id in regex_to_collection_id.items():
        if re.match(regex, s3_uri):
            return collection_id
    return None


def path_to_item(path: str, bucket='imos-data') -> pystac.Item:
    """
    Create a STAC item from an S3 path.
    Args:
        path (str): The S3 path to the file.
        bucket (str, optional): The S3 bucket name. Defaults to 'imos-data'.
    Returns:
        pystac.Item: The STAC item.
    Raises:
        ValueError: If the file type is not supported.
    """
    collection_id = get_collection_id(path)
    if collection_id is None:
        raise ValueError('Could not determine collection ID from path ' + path)

    file_type = path.split('.')[-1]
    match file_type:
        case 'nc':
            result = nc_to_item(f'{bucket}/{path}', collection_id)
        case _:
            raise ValueError(f'Unsupported file type: {file_type}')

    return result


def get_collection_info(collection_id: str) -> dict:
    """
    Query the AODN geonetwork for collection metadata.
    Args:
        collection_id (str): The collection ID.
    Returns:
        dict: A dictionary with title, description, and parent. Each of these
        fields may be None if the metadata is not found.
    """

    endpoint = (
        f"https://catalogue.aodn.org.au/geonetwork/srv/api/0.1/records/{collection_id}"
    )
    headers = {'accept': 'application/json'}
    metadata = requests.get(endpoint, headers=headers).json()

    result = dict()

    result['title'] = (
        metadata.get("mdb:identificationInfo", {})
        .get("mri:MD_DataIdentification", {})
        .get("mri:citation", {})
        .get("cit:CI_Citation", {})
        .get("cit:title", {})
        .get("gco:CharacterString", {})
        .get("#text", None)
    )
    result['description'] = (
        metadata.get("mdb:identificationInfo", {})
        .get("mri:MD_DataIdentification", {})
        .get("mri:abstract", {})
        .get("gco:CharacterString", {})
        .get("#text", None)
    )
    result['parent'] = metadata.get("mdb:parentMetadata", {}).get("@uuidref", None)

    return result

def get_parent(catalog: pystac.Catalog, pystac_object: pystac.STACObject) -> pystac.Catalog:
    """
    Get the parent catalog of a STAC object.
    Args:
        catalog (pystac.Catalog): The STAC catalog.
        pystac_object (pystac.STACObject): The STAC object.
    Returns:
        pystac.Catalog: The parent catalog.
    """
    for link in pystac_object.links:
        if link.rel == 'parent':
            link.resolve_stac_object(catalog)
            return link.target
    return None

def add_to_catalog(item: pystac.Item, catalog: pystac.Catalog):
    """
    Add an item to a STAC catalog, updating its parent collections.
    Args:
        item (pystac.Item): The STAC item to be added.
        catalog (pystac.Catalog): The root STAC catalog.
    """

    collection = get_collection_or_create(catalog, item.collection_id)

    # Check if the item is already in the collection
    if collection.get_item(item.id) is not None:
        # Remove the old item before adding the new one
        # TODO: use versioning extension to deprecate the old item
        collection.remove_item(item.id)
    logger.info(f'Adding item {item.id} to collection {collection.id}')
    collection.add_item(item)
    # Update extents of this collection and parent collections
    updating = collection
    while isinstance(updating, pystac.Collection):
        # will stop at the root catalog
        old_extent = updating.extent
        logger.debug(f'Updating extent of {updating.id}')
        updating.update_extent_from_items()
        if updating.extent == old_extent:
            # No need to update parents if the extent hasn't changed
            logger.debug(f'Extent of {updating.id} has not changed')
            break
        updating = get_parent(catalog, updating)


def get_collection_or_create(
    catalog: pystac.Catalog, collection_id: str
) -> pystac.Collection:
    """
    Get a collection from a catalog, creating it if it does not exist.
    Uses get_collection_info to get information from the AODN geonetwork
    if the collection does not exist.
    Args:
        catalog (pystac.Catalog): The STAC catalog.
        collection_id (str): The collection ID.
    Returns:
        pystac.Collection: The collection.
    """
    # This could be slow for a large catalog with nested collections
    collection = catalog.get_child(collection_id, recursive=True)
    if collection is None:
        logger.info(f'Creating collection {collection_id}')
        metadata = get_collection_info(collection_id)
        collection = pystac.Collection(
            id=collection_id,
            title=metadata['title'],
            description=metadata['description'],
            extent=None, # Will be updated when items are added
        )
        if metadata['parent'] is not None:
            parent_collection = get_collection_or_create(catalog, metadata['parent'])
            parent_collection.add_child(collection)
        else:
            catalog.add_child(collection)
    return collection


def collection_from_items(
    items: List[pystac.Item],
    title: str = None,
    description: str = None,
    license: str = 'proprietary',
    keywords: List[str] = ['oceanography', 'IMOS'],
) -> pystac.Collection:
    """
    Creates a STAC Collection from a list of STAC Items.

    Args:
        items (List[pystac.Item]): A list of STAC Items to be included in the collection.
        title (str, optional): The title of the collection.
        description (str, optional): The description of the collection.
        license (str, optional): The license of the collection. Default is 'proprietary'.
        keywords (List[str], optional): A list of keywords for the collection. Default is ['oceanography', 'IMOS'].

    Returns:
        pystac.Collection: A STAC Collection containing the provided items.

    Notes:
        - The collection ID is derived from the collection_id of the first item in the list.
        - The provider is hardcoded to IMOS.
    """
    # TODO: what's the license for IMOS data?

    collection = pystac.Collection(
        id=items[0].collection_id,
        description=description,
        title=title,
        extent=pystac.Extent.from_items(items),
        license=license,
        providers=[
            pystac.Provider(
                name='IMOS',
                roles=['producer', 'licensor'],
                url='https://imos.org.au/',
            )
        ],
        keywords=keywords,
    )

    # Add all items to the collection
    for item in items:
        collection.add_item(item)

    # # Compute summary for the collection - we should decide what to summarize
    # summary_fields = []  # List of fields to compute summaries for
    # summarizer = pystac.Summarizer(summary_fields)
    # collection.summaries.update(summarizer.summarize(items))

    return collection


def collection_from_nc_files(
    nc_files: List[str], collection_id: str, *args, **kwargs
) -> pystac.Collection:
    """
    Creates a STAC collection from a list of NetCDF files.
    Args:
        nc_files (List[str]): A list of paths to NetCDF files.
        collection_id (str): The unique identifier for the collection.
        *args: Additional positional arguments, passed to collection_from_items.
        **kwargs: Additional keyword arguments, passed to collection_from_items.
    Returns:
        pystac.Collection: The generated STAC collection.
    """
    logger.info(
        f'Creating collection {
                collection_id} from {len(nc_files)} files'
    )
    items = [nc_to_item(nc_file, collection_id) for nc_file in nc_files]
    return collection_from_items(items, *args, **kwargs)


def index_files_from_prefix(
    prefix: str,
    bucket: str = 'imos-data',
    catalog_href: str = 'data_index/catalog.json',
):
    """
    Index files from an S3 prefix.
    Args:
        prefix (str): The S3 prefix to index. e.g. 'IMOS/ANMN/NSW'
        bucket (str, optional): The S3 bucket name. Defaults to 'imos-data'.
        catalog_href (str, optional): The file path to the catalog JSON file. Defaults to 'data_index/catalog.json'.
    """

    catalog = pystac.Catalog.from_file(catalog_href)
    s3 = boto3.resource('s3')
    s3_bucket = s3.Bucket(bucket)
    processed = 0
    for obj in s3_bucket.objects.filter(Prefix=prefix):
        checksum = obj.e_tag.strip('"')

        item_id = obj.key.split('/')[-1].split('.')[0]
        try:
            collection_id = get_collection_id(obj.key)
            if collection_id is None:
                logger.warning(f'Could not determine collection ID for {obj.key}')
                continue
            collection = get_collection_or_create(catalog, collection_id)
            # check if the item is already in the collection
            old_item = collection.get_item(item_id)
            if old_item is not None:
                # check if the checksum matches
                old_checksum = old_item.assets['data'].ext.file.checksum
                if old_checksum == checksum:
                    logger.info(f'Item {item_id} is already indexed')
                    continue
            item = path_to_item(obj.key, bucket=bucket)
            # Add object info to the data asset
            item.ext.add('file')
            file_stac_extension = pystac.extensions.file.FileExtension.ext(item.assets['data'])
            file_stac_extension.apply(checksum=checksum, size=obj.size)
            processed += 1
            add_to_catalog(item, catalog)
            if processed % 100 == 0:
                logger.info(f'Processed {processed} items, saving catalog')
                catalog.normalize_and_save(catalog_href, skip_unresolved=True)
                catalog = pystac.Catalog.from_file(catalog_href)
        except ValueError as e:
            logger.warning(f'Could not index {obj.key}: {e}')
    logger.info(f'Processed {processed} items, saving catalog')
    catalog.normalize_and_save(catalog_href, skip_unresolved=True)


def test_creation_from_collection():
    """
    Test creating a catalog building a collection from a set of items.
    """
    nc_files = [
        's3://imos-data/IMOS/ANMN/NSW/BMP070/gridded_timeseries/IMOS_ANMN-NSW_TZ_20141118_BMP070_FV02_TEMP-gridded-timeseries_END-20240725_C-20240810.nc',
        's3://imos-data/IMOS/ANMN/NSW/CH070/gridded_timeseries/IMOS_ANMN-NSW_TZ_20090815_CH070_FV02_TEMP-gridded-timeseries_END-20240417_C-20240608.nc',
    ]
    collection_id = '279a50e3-21a5-4590-85a0-71f963efab82'
    collection_title = 'IMOS - Moorings - Gridded time-series product'
    collection_description = '''Integrated Marine Observing System (IMOS) have moorings across both it's National Mooring Network and Deep Water Moorings facilities. The National Mooring Network facility comprises a series of national reference stations and regional moorings designed to monitor particular oceanographic phenomena in Australian coastal ocean waters. The Deep Water Moorings facility (formerly known as the Australian Bluewater Observing System) provides the coordination of national efforts in the sustained observation of open ocean properties with particular emphasis on observations important to climate and carbon cycle studies, with selected moorings from its Deep Water Array sub-facility providing data to this collection.

This collection represents the gridded time-series product of temperature observations, binned to one-hour intervals and interpolated to a fixed set of target depths for each IMOS mooring site. Only good-quality measurements (according to the automated quality-control procedures applied by the National Mooring Network) are included.

The observations were made using a range of temperature loggers, conductivity-temperature-depth (CTD) instruments, water-quality monitors (WQM), and temperature sensors on acoustic Doppler current profilers (ADCPs).'''
    collection = collection_from_nc_files(
        nc_files, collection_id, collection_title, collection_description
    )

    catalog = pystac.Catalog(
        id='data_index',
        description='Index of all AODN data',
        title='AODN Data Index',
    )
    catalog.add_child(collection)
    catalog.normalize_and_save(
        './data_index', catalog_type=pystac.CatalogType.SELF_CONTAINED
    )


def test_creation_from_items():
    """
    Test creating a catalog adding one item at a time.
    """
    catalog = pystac.Catalog(
        id='data_index',
        description='Index of all AODN data',
        title='AODN Data Index',
    )
    catalog.normalize_and_save(
        './data_index', catalog_type=pystac.CatalogType.SELF_CONTAINED
    )

    nc_files = [
        'imos-data/IMOS/ANMN/NSW/BMP070/gridded_timeseries/IMOS_ANMN-NSW_TZ_20141118_BMP070_FV02_TEMP-gridded-timeseries_END-20240725_C-20240810.nc',
        'imos-data/IMOS/ANMN/NSW/CH070/gridded_timeseries/IMOS_ANMN-NSW_TZ_20090815_CH070_FV02_TEMP-gridded-timeseries_END-20240417_C-20240608.nc',
    ]

    for nc_file in nc_files:
        collection_id = get_collection_id(nc_file)
        item = nc_to_item(nc_file, collection_id)
        add_to_catalog(item, catalog)
    catalog.normalize_and_save('./data_index', catalog_type=pystac.CatalogType.SELF_CONTAINED)


if __name__ == '__main__':
    # catalog = pystac.Catalog(
    #     id='data_index',
    #     description='Index of all AODN data',
    #     title='AODN Data Index',
    # )
    # catalog.normalize_and_save(
    #     './data_index', catalog_type=pystac.CatalogType.SELF_CONTAINED
    # )
    index_files_from_prefix('IMOS/ANMN/NRS/')
