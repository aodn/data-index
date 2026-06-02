import numpy
import orjson


def _serialize_with_orjson(data: dict) -> dict:
    """Sanitise numpy/xarray types to native Python primitives via orjson round-trip.
    Faster than recursive Python traversal for large metadata dicts."""
    try:
        return orjson.loads(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))
    except TypeError as exc:
        # Some NetCDF attrs contain lone UTF-16 surrogates that orjson rejects.
        if "str is not valid UTF-8" not in str(exc):
            raise
        return orjson.loads(
            orjson.dumps(
                _sanitize_for_json(data),
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )


def _sanitize_string(data: str) -> str:
    """Replace invalid UTF-8 code points with escaped unicode sequences."""
    return data.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _sanitize_dict_key(data) -> str:
    if isinstance(data, bytes):
        return _sanitize_string(data.decode("utf-8", errors="backslashreplace"))
    if isinstance(data, str):
        return _sanitize_string(data)
    return _sanitize_string(str(data))


def _sanitize_for_json(data):
    """Recursively convert numpy/xarray types to native Python primitives."""
    if isinstance(data, dict):
        return {_sanitize_dict_key(k): _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_json(v) for v in data]
    elif isinstance(data, str):
        return _sanitize_string(data)
    elif isinstance(data, numpy.bool_):
        return bool(data)
    elif isinstance(data, (numpy.integer, numpy.int64, numpy.int32)):
        return int(data)
    elif isinstance(data, (numpy.floating, numpy.float64, numpy.float32)):
        return float(data)
    elif isinstance(data, numpy.ndarray):
        return _sanitize_for_json(data.tolist())
    # Handle cases where attributes might be byte-strings
    elif isinstance(data, bytes):
        return data.decode("utf-8", errors="backslashreplace")
    return data
