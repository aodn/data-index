import numpy
import orjson


def _serialize_with_orjson(data: dict) -> dict:
    """Sanitise numpy/xarray types to native Python primitives via orjson round-trip.
    Faster than recursive Python traversal for large metadata dicts."""
    return orjson.loads(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))


def _sanitize_for_json(data):
    """Recursively convert numpy/xarray types to native Python primitives."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_json(v) for v in data]
    elif isinstance(data, numpy.bool_):
        return bool(data)
    elif isinstance(data, (numpy.integer, numpy.int64, numpy.int32)):
        return int(data)
    elif isinstance(data, (numpy.floating, numpy.float64, numpy.float32)):
        return float(data)
    elif isinstance(data, numpy.ndarray):
        return data.tolist()
    # Handle cases where attributes might be byte-strings
    elif isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore")
    return data
