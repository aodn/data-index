import pathlib
import typing

import pydantic

import data_index.protocols
import data_index.xarray_handle


class LocalFetcher(pydantic.BaseModel):
    type: typing.Literal["local_fetcher"] = pydantic.Field(default="local_fetcher")
    local_version_id: str = pydantic.Field(default="__LOCAL__")

    def object_reference_to_staged_object(
        self, object_reference: data_index.protocols.ObjectReference
    ) -> data_index.protocols.StagedObject | data_index.protocols.DeadLetter:
        try:
            if object_reference.version_id != self.local_version_id:
                raise ValueError(
                    "Invalid local version_id: expected "
                    f"'{self.local_version_id}', got '{object_reference.version_id}'"
                )

            bucket_path = pathlib.Path(object_reference.bucket)
            if not bucket_path.is_absolute():
                raise ValueError(
                    f"Local bucket must be an absolute path: '{object_reference.bucket}'"
                )

            key_path = pathlib.Path(object_reference.key)
            if key_path.is_absolute():
                raise ValueError(
                    f"Local key must be a relative path suffix: '{object_reference.key}'"
                )

            path = bucket_path / key_path
            resolved_bucket = bucket_path.resolve(strict=True)
            resolved_path = path.resolve(strict=False)

            if not resolved_path.is_relative_to(resolved_bucket):
                raise ValueError(
                    f"Local key escapes bucket path: bucket='{bucket_path}', key='{object_reference.key}'"
                )

            if not resolved_path.exists():
                raise FileNotFoundError(f"Local file not found: '{resolved_path}'")

            if not resolved_path.is_file():
                raise ValueError(f"Local path is not a file: '{resolved_path}'")

            return data_index.protocols.StagedObject(
                object_reference=object_reference,
                xarray_handle=data_index.xarray_handle.DiskXarrayHandle(
                    path=resolved_path,
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
