from __future__ import annotations

import pathlib
import typing

import polars
import pydantic


class LocalGlobInventorySource(pydantic.BaseModel):
    """Inventory source that scans local files and emits S3-like object references.

    The `bucket` value is used both as the emitted bucket label and as a path
    anchor for key derivation. If a matched absolute path contains that bucket
    segment, the key is the suffix after it.

    Example:
        - bucket = "imos-data"
        - root_path = "/Volumes/4tb-0"
        - matched file = "/Volumes/4tb-0/imos-data/IMOS/Argo/nmdis/2901615/2901615_prof.nc"
        - emitted row:
            - bucket: "imos-data"
            - key: "IMOS/Argo/nmdis/2901615/2901615_prof.nc"

    If the bucket segment is not present in the file path, key falls back to the
    absolute local file path (legacy behavior).
    """

    type: typing.Literal["local_glob"] = pydantic.Field(default="local_glob")

    root_path: pathlib.Path = pydantic.Field(
        description="Base directory to glob from. Matched files must remain under this root after symlink resolution."
    )
    glob_pattern: str = pydantic.Field(
        description="Glob pattern applied from root_path (for example '**/*_prof.nc')."
    )
    bucket: str = pydantic.Field(
        description=(
            "Logical bucket label and path anchor for key derivation. Leading/trailing '/' are stripped. "
            "When this segment appears in an absolute local path, key is derived from the suffix after it."
        )
    )
    local_version_id: str = pydantic.Field(
        default="__LOCAL__",
        description="Static version_id assigned to each local inventory row.",
    )
    max_files: int | None = pydantic.Field(
        default=None,
        ge=1,
        description="Optional cap on number of emitted rows after key sorting.",
    )

    @pydantic.model_validator(mode="after")
    def _validate_root_path(self) -> typing.Self:
        if not self.root_path.exists():
            raise ValueError(f"root_path does not exist: '{self.root_path}'")
        if not self.root_path.is_dir():
            raise ValueError(f"root_path is not a directory: '{self.root_path}'")
        if not self.bucket.strip("/"):
            raise ValueError("bucket must not be empty")
        return self

    @property
    def _normalized_bucket(self) -> str:
        return self.bucket.strip("/")

    def _derive_key(self, resolved_path: pathlib.Path, root: pathlib.Path) -> str:
        path_parts = resolved_path.parts
        bucket_segment = self._normalized_bucket

        # Prefer key derivation relative to the bucket segment in the absolute path.
        # This maps local paths like /Volumes/.../imos-data/IMOS/... to key IMOS/...
        try:
            bucket_index = path_parts.index(bucket_segment)
        except ValueError:
            bucket_index = -1

        if 0 <= bucket_index < len(path_parts) - 1:
            return pathlib.PurePosixPath(*path_parts[bucket_index + 1 :]).as_posix()

        # Preserve legacy local-glob behavior when bucket segment is not present.
        return str(resolved_path)

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
        bucket = self._normalized_bucket

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
            key = self._derive_key(resolved_path=resolved_path, root=root)
            identity = (bucket, key, self.local_version_id)
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
