MAGIC_NUMBERS: dict[bytes, str] = {
    b"CDF\x01": "NETCDF3_CLASSIC",
    b"CDF\x02": "NETCDF3_64BIT_OFFSET",
    b"CDF\x05": "NETCDF5",
    b"\x89HDF\r\n\x1a\n": "NETCDF4",
}


def format_from_magic(magic: bytes) -> str | None:
    """Return the NetCDF format string for the given magic bytes, or None if unrecognised."""
    for signature, fmt in MAGIC_NUMBERS.items():
        if magic[: len(signature)] == signature:
            return fmt
    return None
