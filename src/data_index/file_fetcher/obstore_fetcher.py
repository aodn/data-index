from __future__ import annotations

import pathlib
import typing

import obstore.store
import pydantic

import data_index.protocols
import data_index.xarray_handle


class ObstoreFetcher(pydantic.BaseModel):
    type: typing.Literal["obstore_fetcher"] = pydantic.Field(default="obstore_fetcher")

    extract_path: pathlib.Path = pydantic.Field(
        default_factory=lambda: pathlib.Path(".extract")
    )
    bucket: str = pydantic.Field(default="imos-data")
    region: str = pydantic.Field(default="ap-southeast-2")
    skip_signature: bool = pydantic.Field(default=True)
    min_chunk_size: int = pydantic.Field(
        default=100 * 1024**2, description="Defaults to 100MB"
    )
    override_downloaded_files: bool = pydantic.Field(default=False)
    _store: obstore.store.S3Store = pydantic.PrivateAttr()

    @pydantic.model_validator(mode="after")
    def _initialize_store(self) -> "ObstoreFetcher":
        self._store = obstore.store.S3Store(
            bucket=self.bucket, region=self.region, skip_signature=self.skip_signature
        )
        return self

    @property
    def store(self) -> obstore.store.S3Store:
        """Expose the store via a read-only property."""
        return self._store

    def object_reference_to_disk_xarray_handle(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ) -> data_index.xarray_handle.DiskXarrayHandle:
        """
        Construct a disk xarray handle utilising the stream to disk utility.
        """

        path = self.stream_to_disk(object_reference=object_reference)
        return data_index.xarray_handle.DiskXarrayHandle(
            path=path,
        )

    def stream_to_disk(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ) -> pathlib.Path:
        """
        Stream the source object to disk in 10MB chunks
        """

        # Construct the write path
        write_path = self.extract_path / object_reference.path

        # Optionally don't re-download existing files
        if write_path.is_file() and not self.override_downloaded_files:
            return write_path

        # Stream the path to disk and return the path
        write_path.parent.mkdir(parents=True, exist_ok=True)
        with open(write_path, "wb") as f:
            for chunk in self.get_stream(object_reference=object_reference):
                f.write(chunk)
        return write_path

    def get_stream(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ):
        """
        Convert an ObjectReference into a stream generator.
        """

        options = dict()

        # If version id
        if object_reference.version_id:
            options.update({"version": object_reference.version_id})

        return self.store.get(
            path=object_reference.key,
            options=options,
        ).stream(min_chunk_size=self.min_chunk_size)

    def fetch(
        self, object_references: list[data_index.protocols.ObjectReference]
    ) -> list[data_index.protocols.ObjectReference]:
        """
        Populate all ObjectReferences with disk xarray handles.

        Causes download of all passed in object_references to `self.extract_path`
        """
        return [
            object_reference.with_xarray_handle(
                xarray_handle=self.object_reference_to_disk_xarray_handle(
                    object_reference=object_reference
                )
            )
            for object_reference in object_references
        ]


if __name__ == "__main__":
    import rich

    obstore_fetcher = ObstoreFetcher()
    rich.print(obstore_fetcher)

    object_reference = data_index.protocols.ObjectReference(
        bucket="imos-data",
        key="IMOS/ANMN/SA/B1/Biogeochem_profiles/IMOS_ANMN-SA_CDEFKOSTUZ_20080212T053920Z_B1_FV01_Profile-SeacatPlus_C-20170214T015514Z.nc",
        version_id=None,
        size=0,
        xarray_handle=None,
    )

    object_references = obstore_fetcher.fetch([object_reference])
    rich.print(object_references)
