from __future__ import annotations

import pathlib
import typing

import polars
import pydantic


class LocalGlobInventorySource(pydantic.BaseModel):
    """Inventory source that scans local files and emits S3-like object references.

    The configured `bucket` is treated as a path anchor segment. For each
    matched file, if that segment appears in the absolute local path, emitted
    `bucket` becomes the full absolute prefix up to and including that segment,
    and emitted `key` is the suffix after it.

    Example:
        - bucket = "imos-data"
        - root_path = "/Volumes/4tb-0"
        - matched file = "/Volumes/4tb-0/imos-data/IMOS/Argo/nmdis/2901615/2901615_prof.nc"
        - emitted row:
            - bucket: "/Volumes/4tb-0/imos-data"
            - key: "IMOS/Argo/nmdis/2901615/2901615_prof.nc"

    If the bucket anchor segment is not present in a matched path, inventory
    fails fast.
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
            "Path anchor segment used to split absolute local file paths into emitted "
            "`bucket` and `key`. Leading/trailing '/' are stripped before matching."
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

    def _derive_bucket_and_key(self, resolved_path: pathlib.Path) -> tuple[str, str]:
        path_parts = resolved_path.parts
        bucket_segment = self._normalized_bucket

        # Use the first matching bucket segment in the absolute path.
        try:
            bucket_index = path_parts.index(bucket_segment)
        except ValueError:
            bucket_index = -1

        if 0 <= bucket_index < len(path_parts) - 1:
            bucket = pathlib.PurePosixPath(*path_parts[: bucket_index + 1]).as_posix()
            key = pathlib.PurePosixPath(*path_parts[bucket_index + 1 :]).as_posix()
            return bucket, key

        raise ValueError(
            "Bucket anchor segment not found in matched path: "
            f"anchor='{bucket_segment}', path='{resolved_path}'"
        )

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
            bucket, key = self._derive_bucket_and_key(resolved_path=resolved_path)
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


if __name__ == "__main__":
    # --- Local inventory + fetch config ---
    LOCAL_ROOT_PATH = pathlib.Path("/Volumes/4tb-0/imos-data/IMOS/Argo/")
    LOCAL_GLOB_PATTERN = "**/*_prof.nc"
    LOCAL_BUCKET = "imos-data"
    LOCAL_VERSION_ID = "__LOCAL__"

    INVENTORY_SOURCE = LocalGlobInventorySource(
        root_path=LOCAL_ROOT_PATH,
        glob_pattern=LOCAL_GLOB_PATTERN,
        bucket=LOCAL_BUCKET,
        local_version_id=LOCAL_VERSION_ID,
        max_files=10,
    )
    print(INVENTORY_SOURCE.inventory())
