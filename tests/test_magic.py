import pytest

from data_index.xarray_handle._magic import MAGIC_NUMBERS, format_from_magic


@pytest.mark.parametrize(
    "magic,expected",
    [
        (b"CDF\x01\x00\x00\x00\x00", "NETCDF3_CLASSIC"),
        (b"CDF\x02\x00\x00\x00\x00", "NETCDF3_64BIT_OFFSET"),
        (b"CDF\x05\x00\x00\x00\x00", "NETCDF5"),
        (b"\x89HDF\r\n\x1a\n", "NETCDF4"),
    ],
)
def test_format_from_magic_recognises_all_known_signatures(magic, expected):
    assert format_from_magic(magic) == expected


def test_format_from_magic_returns_none_for_unknown_bytes():
    assert format_from_magic(b"\x00\x01\x02\x03\x04\x05\x06\x07") is None


def test_magic_numbers_covers_all_common_netcdf_formats():
    formats = set(MAGIC_NUMBERS.values())
    assert {"NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET", "NETCDF5", "NETCDF4"} == formats
