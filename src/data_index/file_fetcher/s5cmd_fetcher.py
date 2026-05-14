import pathlib
import sh
import cloudpathlib
import re
from data_index.protocols import ManifestEntry

class S5CMDFetcher:
    """FileFetcher implementation that downloads files from S3 using S5CMD."""

    def __init__(self):
        self._check_availability()

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
    def _parse_s5cmd_output(stdout_str: str) -> list[ManifestEntry]:
        """
        Parses s5cmd stdout to identify successful downloads.
        Expected line format: cp s3://bucket/key local/path
        """
        entries = []
        # Regex to capture the S3 URI and the Local Path from 'cp' lines
        # s5cmd output usually looks like: cp s3://source local/dest
        pattern = re.compile(r"cp\s+(s3://\S+)\s+(\S+)")

        for line in stdout_str.splitlines():
            match = pattern.search(line)
            if match:
                s3_uri = match.group(1)
                local_path = pathlib.Path(match.group(2)).resolve()
                entries.append(ManifestEntry(s3_uri=s3_uri, absolute_path=local_path))
        
        return entries

    def fetch(self, uris: list[str], extract_path: pathlib.Path) -> list[ManifestEntry]:
        if not uris:
            return []

        commands = self._prepare_commands(uris, extract_path)
        input_stream = "\n".join(commands) + "\n"
        
        try:
            # Capture stdout to see what s5cmd actually did
            output = sh.s5cmd("run", _in=input_stream)
            return self._parse_s5cmd_output(stdout_str=output)
        except sh.ErrorReturnCode as e:
            # Even on partial failure, s5cmd might have downloaded some files
            # You could choose to parse e.stdout here if needed
            error_msg = e.stderr.decode() if e.stderr else e.stdout.decode()
            raise RuntimeError(f"s5cmd execution failed:\n{error_msg}") from e