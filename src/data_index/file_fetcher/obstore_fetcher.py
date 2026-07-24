from __future__ import annotations

import concurrent.futures
import pathlib
import typing

import obstore.store
import pydantic

import data_index.protocols
import data_index.xarray_handle

if typing.TYPE_CHECKING:
    from obstore import GetOptions


class ObstoreFetcher(pydantic.BaseModel):
    type: typing.Literal["obstore_fetcher"] = pydantic.Field(default="obstore_fetcher")

    extract_path: pathlib.Path = pydantic.Field(default=pathlib.Path(".extract"))
    bucket: str = pydantic.Field(default="imos-data")
    region: str = pydantic.Field(default="ap-southeast-2")
    skip_signature: bool = pydantic.Field(
        default=True, description="Whether to sign the S3 requests"
    )
    min_chunk_size: int = pydantic.Field(
        default=100 * 1024**2, description="Defaults to 100MiB"
    )
    override_downloaded_files: bool = pydantic.Field(
        default=False,
        description="Whether to overwrite existing files. Irrelevant for ephemeral compute",
    )
    _store: obstore.store.S3Store = pydantic.PrivateAttr()

    @pydantic.model_validator(mode="after")
    def _initialize_store(self) -> ObstoreFetcher:
        self._store = obstore.store.S3Store(
            bucket=self.bucket, region=self.region, skip_signature=self.skip_signature
        )
        return self

    @property
    def store(self) -> obstore.store.S3Store:
        """Expose the store via a read-only property."""
        return self._store

    def object_reference_to_staged_object(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ) -> data_index.protocols.StagedObject | data_index.protocols.DeadLetter:
        """
        Construct a disk xarray handle utilising the stream to disk utility.
        """

        # Try to construct a disk xarray handle for the object reference
        try:
            path = self.stream_to_disk(object_reference=object_reference)
            return data_index.protocols.StagedObject(
                object_reference=object_reference,
                xarray_handle=data_index.xarray_handle.DiskXarrayHandle(
                    path=path,
                ),
            )
        except Exception as e:
            return data_index.protocols.DeadLetter.from_object_reference(
                object_reference=object_reference, error=str(e)
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
            f.writelines(self.get_stream(object_reference=object_reference))
        return write_path

    def get_stream(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ):
        """
        Convert an ObjectReference into a stream generator.
        """

        options: GetOptions = {}

        # If version id
        if object_reference.version_id:
            options.update({"version": object_reference.version_id})

        return self.store.get(
            path=object_reference.key,
            options=options,
        ).stream(min_chunk_size=self.min_chunk_size)

    def fetch(
        self, object_references: list[data_index.protocols.ObjectReference]
    ) -> tuple[
        list[data_index.protocols.StagedObject], list[data_index.protocols.DeadLetter]
    ]:
        """
        Populate all ObjectReferences with disk xarray handles.

        Causes download of all passed in object_references to `self.extract_path`
        """

        staged_objects = [
            self.object_reference_to_staged_object(object_reference=object_reference)
            for object_reference in object_references
        ]

        return (
            [
                staged_object
                for staged_object in staged_objects
                if isinstance(staged_object, data_index.protocols.StagedObject)
            ],
            [
                staged_object
                for staged_object in staged_objects
                if isinstance(staged_object, data_index.protocols.DeadLetter)
            ],
        )


class ConcurrentObstoreFetcher(ObstoreFetcher):
    type: typing.Literal["concurrent_obstore_fetcher"] = pydantic.Field(
        default="concurrent_obstore_fetcher"
    )

    max_workers: int = pydantic.Field(
        default=8,
        description="Max concurrency of the file fetching. Ensure this is lower than available threads in the flow task runner",
    )

    def fetch(
        self, object_references: list[data_index.protocols.ObjectReference]
    ) -> tuple[
        list[data_index.protocols.StagedObject], list[data_index.protocols.DeadLetter]
    ]:
        """
        Populate all ObjectReferences with disk xarray handles.

        Causes download of all passed in object_references to `self.extract_path`
        """

        # Concurrently retrieve objects
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(
                    self.object_reference_to_staged_object, object_reference
                )
                for object_reference in object_references
            ]

        # Collect objects
        staged_objects = [future.result() for future in futures]

        # Return sorted StagedObject
        return (
            [
                staged_object
                for staged_object in staged_objects
                if isinstance(staged_object, data_index.protocols.StagedObject)
            ],
            [
                staged_object
                for staged_object in staged_objects
                if isinstance(staged_object, data_index.protocols.DeadLetter)
            ],
        )
