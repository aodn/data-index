UNKNOWN_FACILITY = "UNKNOWN"


def derive_facility(key: str) -> str:
    """Derive facility as the second key segment.

    E.g. ``IMOS/ANMN/NSW/file.nc`` -> ``ANMN``.
    Missing/invalid keys are coerced to ``UNKNOWN``.
    """
    parts = [part for part in key.split("/") if part]
    if len(parts) < 3:
        return UNKNOWN_FACILITY
    facility = parts[1].strip()
    return facility if facility else UNKNOWN_FACILITY


def derive_collection(s3_uri: str) -> str | None:
    """Backward-compatible adapter to legacy collection semantics."""
    parts = s3_uri.split("/", 3)
    key = parts[3] if len(parts) > 3 else ""
    facility = derive_facility(key)
    return None if facility == UNKNOWN_FACILITY else facility
