def derive_collection(s3_uri: str) -> str | None:
    """Return the second path segment of the S3 key as the collection identifier.

    E.g. 's3://bucket/IMOS/ANMN/NSW/file.nc' → 'ANMN'.
    Returns None if the URI has fewer than two path segments (avoids treating
    a filename as a collection for shallow URIs like 's3://bucket/IMOS/file.nc').
    """
    parts = s3_uri.split("/")
    # parts: ['s3:', '', 'bucket', 'seg1', 'seg2', ...]
    #                               idx3    idx4
    # Require at least one segment beyond the collection (len > 5) so that
    # a file sitting exactly one level deep doesn't become its own collection.
    return parts[4] if len(parts) > 5 else None
