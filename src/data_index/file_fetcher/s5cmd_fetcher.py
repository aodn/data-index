import pathlib
import logging
import sh
import cloudpathlib
import re
from data_index.protocols import BatchEntry, XarrayHandle
from data_index.xarray_handle.disk_xarray_handle import DiskXarrayHandle
import pydantic

logger = logging.getLogger(__name__)


class S5CMDFetcher(pydantic.BaseModel):
    """FileFetcher implementation that downloads files from S3 using S5CMD."""

    extract_path: pathlib.Path = pydantic.Field(
        default_factory=lambda: pathlib.Path("/tmp")
    )
    num_workers: int = pydantic.Field(default=256)
    anon: bool = pydantic.Field(default=False)

    @staticmethod
    def _check_availability() -> str:
        try:
            return str(sh.s5cmd("version")).strip()
        except sh.CommandNotFound:
            raise RuntimeError("s5cmd not found in PATH.")

    @staticmethod
    def _prepare_commands(uris: list[str], extract_path: pathlib.Path) -> list[str]:
        commands = []
        for uri in uris:
            s3_path = cloudpathlib.S3Path(uri)
            local_path = extract_path / s3_path.bucket / s3_path.key
            commands.append(f"cp {uri} {local_path}")
        return commands

    @staticmethod
    def _parse_s5cmd_output(stdout_str: str) -> list[XarrayHandle]:
        """
        Parses s5cmd stdout to identify successful downloads.
        Expected line format: cp s3://bucket/key local/path
        """
        entries = []
        pattern = re.compile(r"cp\s+(s3://\S+)\s+(\S+)")

        for line in stdout_str.splitlines():
            match = pattern.search(line)
            if match:
                s3_uri = match.group(1)
                local_path = pathlib.Path(match.group(2)).resolve()
                entries.append(DiskXarrayHandle(path=local_path, s3_uri=s3_uri))

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

        uris = [entry.uri for entry in entries]
        if not uris:
            return []

        commands = self._prepare_commands(uris, self.extract_path)
        input_stream = "\n".join(commands) + "\n"

        try:
            # Capture stdout to see what s5cmd actually did
            args = ["--numworkers", self.num_workers]
            if self.anon:
                args.append("--no-sign-request")
            args += ["run"]
            output = sh.s5cmd(*args, _in=input_stream)
            return self._parse_s5cmd_output(stdout_str=self._decode_stream(output))
        except sh.ErrorReturnCode as e:
            stdout_str = self._decode_stream(e.stdout)
            stderr_str = self._decode_stream(e.stderr)

            manifest = self._parse_s5cmd_output(stdout_str=stdout_str)
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
