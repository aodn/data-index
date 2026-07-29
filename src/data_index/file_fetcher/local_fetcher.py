import pathlib
import typing

import pydantic

import data_index.protocols
import data_index.xarray_handle


class LocalFetcher(pydantic.BaseModel):
    type: typing.Literal["local_fetcher"] = pydantic.Field(default="local_fetcher")
    bucket: str = pydantic.Field(
        description="Logical local source label expected on ObjectReference.bucket"
    )
    local_version_id: str = pydantic.Field(default="__LOCAL__")

    def object_reference_to_staged_object(
        self, object_reference: data_index.protocols.ObjectReference
    ) -> data_index.protocols.StagedObject | data_index.protocols.DeadLetter:
        try:
            if object_reference.bucket != self.bucket:
                raise ValueError(
                    f"Bucket mismatch: expected '{self.bucket}', got '{object_reference.bucket}'"
                )

            if object_reference.version_id != self.local_version_id:
                raise ValueError(
                    "Invalid local version_id: expected "
                    f"'{self.local_version_id}', got '{object_reference.version_id}'"
                )

            path = pathlib.Path(object_reference.key)

            if not path.is_absolute():
                raise ValueError(
                    f"Local key must be an absolute file path: '{object_reference.key}'"
                )

            if not path.exists():
                raise FileNotFoundError(f"Local file not found: '{path}'")

            if not path.is_file():
                raise ValueError(f"Local path is not a file: '{path}'")

            return data_index.protocols.StagedObject(
                object_reference=object_reference,
                xarray_handle=data_index.xarray_handle.DiskXarrayHandle(
                    path=path,
                    delete_on_cleanup=False,
                ),
            )
        except Exception as e:
            return data_index.protocols.DeadLetter.from_object_reference(
                object_reference=object_reference,
                error=str(e),
            )

    def fetch(
        self, object_references: list[data_index.protocols.ObjectReference]
    ) -> tuple[
        list[data_index.protocols.StagedObject], list[data_index.protocols.DeadLetter]
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
