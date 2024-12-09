import logging
import re

import boto3
import pystac
import requests
from nc_to_stac import nc_to_item

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def join_extents(extent1: pystac.Extent, extent2: pystac.Extent) -> pystac.Extent:
    """
    Join two extents, taking the union of their bounding boxes and temporal intervals.
    Args:
        extent1 (pystac.Extent): The first extent.
        extent2 (pystac.Extent): The second extent.
    Returns:
        pystac.Extent: The joined extent.
    """
    if extent1 is None:
        return extent2

    bboxes = extent1.spatial.bboxes + extent2.spatial.bboxes
    merged_bboxes = [
        [
            min([bbox[0] for bbox in bboxes]),
            min([bbox[1] for bbox in bboxes]),
            max([bbox[2] for bbox in bboxes]),
            max([bbox[3] for bbox in bboxes]),
        ]
    ]
    merged_extra = extent1.spatial.extra_fields
    merged_extra.update(extent2.spatial.extra_fields)
    spatial = pystac.SpatialExtent(merged_bboxes, merged_extra)
    intervals = extent1.temporal.intervals + extent2.temporal.intervals
    merged_intervals = [
        [
            min([interval[0] for interval in intervals]),
            max([interval[1] for interval in intervals]),
        ]
    ]
    merged_extra = extent1.temporal.extra_fields
    merged_extra.update(extent2.temporal.extra_fields)
    temporal = pystac.TemporalExtent(merged_intervals, merged_extra)

    return pystac.Extent(spatial, temporal)


class AODNDataIndexer:
    """
    A class to index AODN data into a STAC catalog.
    """

    def __init__(
        self, catalog_href: str = 'examples/data_index/catalog.json', create=False
    ):
        """
        Initialize the indexer.
        Args:
            catalog_href (str, optional): The path to the catalog file. Defaults to 'data_index/catalog.json'.
            create (bool, optional): Whether to create a new catalog. Defaults to False.
        """
        self.catalog_href = catalog_href
        if create:
            self.catalog = pystac.Catalog(
                id='data_index',
                description='Index of all AODN data',
                title='AODN Data Index',
            )
            self.save_catalog()
        else:
            self.catalog = pystac.Catalog.from_file(catalog_href)

    def save_catalog(self, skip_unresolved=True):
        """
        Save the catalog to disk.
        Args:
            skip_unresolved (bool, optional): Whether to skip unresolved links. Defaults to True.
        """
        self.catalog.normalize_and_save(
            self.catalog_href,
            catalog_type=pystac.CatalogType.SELF_CONTAINED,
            skip_unresolved=skip_unresolved,
        )

    def reload_catalog(self):
        """
        Reload the catalog from disk.
        """
        self.catalog = pystac.Catalog.from_file(self.catalog_href)

    @staticmethod
    def path_to_item(path: str, bucket='imos-data', item_id=None) -> pystac.Item:
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
        collection_id = AODNDataIndexer.get_collection_id(path)
        if collection_id is None:
            raise ValueError('Could not determine collection ID from path ' + path)

        file_type = path.split('.')[-1]
        match file_type:
            case 'nc':
                result = nc_to_item(f'{bucket}/{path}', collection_id, item_id)
            case _:
                raise ValueError(f'Unsupported file type: {file_type}')

        return result

    def add_to_catalog(self, item: pystac.Item):
        """
        Add an item to a STAC catalog, updating its parent collections.
        Args:
            item (pystac.Item): The STAC item to be added.
        """

        collection = self.get_collection_or_create(item.collection_id)

        removed = False
        # Check if the item is already in the collection
        if collection.get_item(item.id) is not None:
            # Remove the old item before adding the new one
            # TODO: use versioning extension to deprecate the old item
            import pdb; pdb.set_trace()
            logger.error(f'Removing item {item.id} from collection {collection.id}')
            collection.remove_item(item.id)
            removed = True
        logger.debug(f'Adding item {item.id} to collection {collection.id}')
        collection.add_item(item)
        # Update extents of this collection and parent collections
        updating = collection
        while isinstance(updating, pystac.Collection):
            # will stop at the root catalog
            old_extent = updating.extent
            logger.debug(f'Updating extent of {updating.id}')
            if removed:
                updating.update_extent_from_items()
            else:
                item_extent = pystac.Extent.from_items([item])
                updating.extent = join_extents(updating.extent, item_extent)
            if updating.extent == old_extent:
                # No need to update parents if the extent hasn't changed
                logger.debug(f'Extent of {updating.id} has not changed')
                break
            updating = updating.get_parent()

    def get_collection_or_create(self, collection_id: str) -> pystac.Collection:
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
        collection = self.catalog.get_child(collection_id, recursive=True)
        if collection is None:
            logger.info(f'Creating collection {collection_id}')
            metadata = self.get_collection_info(collection_id)
            collection = pystac.Collection(
                id=collection_id,
                title=metadata['title'] or collection_id,
                description=metadata['description'],
                extent=None,  # Will be updated when items are added
                # We could add license, providers, keywords, etc. here
            )
            if metadata['parent'] is not None:
                parent_collection = self.get_collection_or_create(metadata['parent'])
                parent_collection.add_child(collection)
            else:
                self.catalog.add_child(collection)
        return collection

    def index_files_from_prefix(self, prefix: str, bucket: str = 'imos-data'):
        """
        Index files from an S3 prefix.
        Args:
            prefix (str): The S3 prefix to index. e.g. 'IMOS/ANMN/NSW'
            bucket (str, optional): The S3 bucket name. Defaults to 'imos-data'.
        """

        s3 = boto3.resource('s3')
        s3_bucket = s3.Bucket(bucket)
        processed = 0
        for obj in s3_bucket.objects.filter(Prefix=prefix):
            checksum = obj.e_tag.strip('"')

            item_id = obj.key.replace('/', '_').replace('.', '_')
            try:
                collection_id = AODNDataIndexer.get_collection_id(obj.key)
                if collection_id is None:
                    logger.warning(f'Could not determine collection ID for {obj.key}')
                    continue
                collection = self.get_collection_or_create(collection_id)
                # check if the item is already in the collection
                old_item = collection.get_item(item_id)
                if old_item is not None:
                    # check if the checksum matches
                    old_checksum = old_item.assets['data'].ext.file.checksum
                    if old_checksum == checksum:
                        logger.info(f'Item {item_id} is already indexed')
                        continue
                item = AODNDataIndexer.path_to_item(obj.key, bucket=bucket, item_id=item_id)
                # Add object info to the data asset
                item.ext.add('file')
                file_stac_extension = pystac.extensions.file.FileExtension.ext(
                    item.assets['data']
                )
                file_stac_extension.apply(checksum=checksum, size=obj.size)
                processed += 1
                self.add_to_catalog(item)
                if processed % 100 == 0:
                    logger.info(f'Processed {processed} items, saving catalog')
                    self.save_catalog()
                    self.reload_catalog()
            except ValueError as e:
                logger.warning(f'Could not index {obj.key}: {e}')
        logger.info(f'Processed {processed} items, saving catalog')
        self.save_catalog()

    @staticmethod
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
            "^IMOS/ANMN/.*/hourly_timeseries/.*": "efd8201c-1eca-412e-9ad2-0534e96cea14",
            "^IMOS/ANMN/.*/gridded_timeseries/.*": "279a50e3-21a5-4590-85a0-71f963efab82",
            "^IMOS/ANMN/.*/aggregated_timeseries/.*": "moorings-aggregated-timeseries-product",
        }
        for regex, collection_id in regex_to_collection_id.items():
            if re.match(regex, s3_uri):
                return collection_id
        return None

    @staticmethod
    def get_collection_info(collection_id: str) -> dict:
        """
        Query the AODN geonetwork for collection metadata.
        Args:
            collection_id (str): The collection ID.
        Returns:
            dict: A dictionary with title, description, and parent. Each of these
            fields may be None if the metadata is not found.
        """

        endpoint = f"https://catalogue.aodn.org.au/geonetwork/srv/api/0.1/records/{collection_id}"
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


if __name__ == '__main__':
    # indexer = AODNDataIndexer(catalog_href='examples/data_index2/catalog.json', create=True)
    indexer = AODNDataIndexer(
        catalog_href='examples/data_index/catalog.json', 
        # create=True
    )
    indexer.index_files_from_prefix('IMOS/ANMN/NRS/')
