from __future__ import annotations

import pathlib
import typing

import polars
import pydantic


class LocalGlobInventorySource(pydantic.BaseModel):
    type: typing.Literal["local_glob"] = pydantic.Field(default="local_glob")

    root_path: pathlib.Path
    glob_pattern: str
    bucket: str
    local_version_id: str = pydantic.Field(default="__LOCAL__")
    max_files: int | None = pydantic.Field(default=None, ge=1)

    @pydantic.model_validator(mode="after")
    def _validate_root_path(self) -> typing.Self:
        if not self.root_path.exists():
            raise ValueError(f"root_path does not exist: '{self.root_path}'")
        if not self.root_path.is_dir():
            raise ValueError(f"root_path is not a directory: '{self.root_path}'")
        return self

    @staticmethod
    def _empty_inventory() -> polars.DataFrame:
        return polars.DataFrame(
            schema={
                "bucket": polars.String,
                "key": polars.String,
                "version_id": polars.String,
                "size": polars.Int64,
            }
        )

    def inventory(self) -> polars.DataFrame:
        root = self.root_path.resolve(strict=True)

        identities: dict[tuple[str, str, str], int] = {}

        for candidate in self.root_path.glob(self.glob_pattern):
            if not candidate.is_file():
                continue

            resolved_path = candidate.resolve(strict=True)

            if not resolved_path.is_relative_to(root):
                raise ValueError(
                    f"Matched path escapes root_path via symlink: '{candidate}' -> '{resolved_path}'"
                )

            size_bytes = resolved_path.stat().st_size
            key = str(resolved_path)
            identity = (self.bucket, key, self.local_version_id)
            identities[identity] = size_bytes

        if not identities:
            return self._empty_inventory()

        rows = [
            {
                "bucket": bucket,
                "key": key,
                "version_id": version_id,
                "size": size,
            }
            for (bucket, key, version_id), size in identities.items()
        ]
        rows.sort(key=lambda row: row["key"])

        if self.max_files is not None:
            rows = rows[: self.max_files]

        return polars.DataFrame(
            data=rows,
            schema={
                "bucket": polars.String,
                "key": polars.String,
                "version_id": polars.String,
                "size": polars.Int64,
            },
        )
