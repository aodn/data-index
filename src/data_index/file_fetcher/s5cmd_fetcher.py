import hashlib
import logging
import pathlib
import re
import typing

import pydantic
import sh

from data_index.protocols import BatchEntry, ObjectReference, XarrayHandle
from data_index.xarray_handle.disk_xarray_handle import DiskXarrayHandle

logger = logging.getLogger(__name__)


class S5CMDFetcher(pydantic.BaseModel):
    """FileFetcher implementation that downloads files from S3 using S5CMD."""

    type: typing.Literal["s5cmd_fetcher"] = pydantic.Field(default="s5cmd_fetcher")

    extract_path: pathlib.Path = pydantic.Field(
        default_factory=lambda: pathlib.Path("/tmp")
    )
    num_workers: int = pydantic.Field(default=256)
    anon: bool = pydantic.Field(default=False)

    @staticmethod
    def _version_path_token(version_id: str) -> str:
        """Filesystem-safe token for a version ID."""
        return hashlib.sha256(version_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _quote_version_id(version_id: str) -> str:
        return version_id.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _check_availability() -> str:
        try:
            return str(sh.s5cmd("version")).strip()
        except sh.CommandNotFound:
            raise RuntimeError("s5cmd not found in PATH.")

    @staticmethod
    def _prepare_commands(
        entries: list[BatchEntry], extract_path: pathlib.Path
    ) -> tuple[list[str], dict[str, ObjectReference]]:
        commands = []
        destination_object_refs: dict[str, ObjectReference] = {}
        for entry in entries:
            source_uri = entry.object_ref.as_uri()
            quoted_version_id = S5CMDFetcher._quote_version_id(
                entry.object_ref.version_id
            )
            version_token = S5CMDFetcher._version_path_token(
                entry.object_ref.version_id
            )
            local_path = (
                extract_path
                / entry.object_ref.bucket
                / version_token
                / entry.object_ref.key
            )
            local_path_str = str(local_path.resolve())
            destination_object_refs[local_path_str] = entry.object_ref
            commands.append(
                f'cp --version-id "{quoted_version_id}" {source_uri} {local_path}'
            )
        return commands, destination_object_refs

    @staticmethod
    def _parse_s5cmd_output(
        stdout_str: str, destination_object_refs: dict[str, ObjectReference]
    ) -> list[XarrayHandle]:
        """
        Parses s5cmd stdout to identify successful downloads.
        Expected line format: cp s3://bucket/key local/path
        """
        entries = []
        pattern = re.compile(r'cp(?:\s+--version-id\s+"[^"]*")?\s+(s3://\S+)\s+(\S+)')

        for line in stdout_str.splitlines():
            match = pattern.search(line)
            if match:
                s3_uri = match.group(1)
                local_path = pathlib.Path(match.group(2)).resolve()
                object_ref = destination_object_refs.get(str(local_path))
                if object_ref is None:
                    try:
                        object_ref = ObjectReference.from_s3_uri(s3_uri)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Unable to resolve object reference for download: {line}"
                        ) from exc
                entries.append(
                    DiskXarrayHandle(
                        path=local_path,
                        object_ref=object_ref,
                    )
                )

        return entries

    @staticmethod
    def _decode_stream(stream: object) -> str:
        if stream is None:
            return ""
        if isinstance(stream, bytes):
            return stream.decode(errors="replace")
        return str(stream)

    @staticmethod
    def _is_missing_key_error(line: str) -> bool:
        return "NoSuchKey" in line or "The specified key does not exist" in line

    def fetch(self, entries: list[BatchEntry]) -> list[XarrayHandle]:

        self._check_availability()

        if not entries:
            return []

        commands, destination_object_refs = self._prepare_commands(
            entries, self.extract_path
        )
        input_stream = "\n".join(commands) + "\n"

        try:
            # Capture stdout to see what s5cmd actually did
            args = ["--numworkers", self.num_workers]
            if self.anon:
                args.append("--no-sign-request")
            args += ["run"]
            output = sh.s5cmd(*args, _in=input_stream)
            return self._parse_s5cmd_output(
                stdout_str=self._decode_stream(output),
                destination_object_refs=destination_object_refs,
            )
        except sh.ErrorReturnCode as e:
            stdout_str = self._decode_stream(e.stdout)
            stderr_str = self._decode_stream(e.stderr)

            manifest = self._parse_s5cmd_output(
                stdout_str=stdout_str,
                destination_object_refs=destination_object_refs,
            )
            error_lines = [
                line.strip() for line in stderr_str.splitlines() if line.strip()
            ]
            missing_key_lines = [
                line for line in error_lines if self._is_missing_key_error(line)
            ]
            other_error_lines = [
                line for line in error_lines if not self._is_missing_key_error(line)
            ]

            if missing_key_lines:
                logger.warning(
                    "s5cmd skipped missing S3 objects (%d):\n%s",
                    len(missing_key_lines),
                    "\n".join(missing_key_lines),
                )

            if other_error_lines:
                raise RuntimeError(
                    f"s5cmd execution failed:\n{'\n'.join(other_error_lines)}"
                ) from e

            return manifest
