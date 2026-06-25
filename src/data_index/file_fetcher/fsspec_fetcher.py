import typing

import pydantic

import data_index.protocols
import data_index.xarray_handle


class FSSpecFetcher(pydantic.BaseModel):
    """
    FileFetcher implementation that downloads files from S3 using boto3.

    Note this fetcher does not actually load any data.

    It passes back handles that intelligently query header information from NetCDF files in Cloud.
    """

    type: typing.Literal["fsspec_fetcher"] = pydantic.Field(default="fsspec_fetcher")

    block_size: int = pydantic.Field(default=1024**2)

    def object_reference_to_staged_object(
        self,
        object_reference: data_index.protocols.ObjectReference,
    ) -> data_index.protocols.StagedObject | data_index.protocols.DeadLetter:
        # Try to construct a fsspec xarray handle for the object reference
        try:
            path = self.stream_to_disk(object_reference=object_reference)
            return data_index.protocols.StagedObject(
                object_reference=object_reference,
                xarray_handle=data_index.xarray_handle.FSSpecXarrayHandle(
                    path=path,
                ),
            )
        except Exception as e:
            return data_index.protocols.DeadLetter.from_object_reference(
                object_reference=object_reference, error=str(e)
            )

    def fetch(
        self, object_references: list[data_index.protocols.ObjectReference]
    ) -> tuple[
        list[data_index.protocols.StagedObject, list[data_index.protocols.DeadLetter]]
    ]:

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
