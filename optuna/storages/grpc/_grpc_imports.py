from optuna._imports import try_import


with try_import() as _imports:
    from google.protobuf import descriptor
    from google.protobuf import descriptor_pool
    from google.protobuf import runtime_version
    from google.protobuf import symbol_database
    from google.protobuf.internal import builder
    import grpc
    from grpc import __version__ as grpc_version
    from grpc._utilities import first_version_is_lower

    GRPC_GENERATED_VERSION = "1.68.1"
    GRPC_VERSION = grpc_version
    _version_not_supported = False

    try:
        _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
    except ImportError:
        _version_not_supported = True

    if _version_not_supported:
        raise RuntimeError(
            f"The grpc package installed is at version {GRPC_VERSION},"
            + " but the generated code in _api_pb2_grpc.py depends on"
            + f" grpcio>={GRPC_GENERATED_VERSION}."
            + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
            + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
        )


__all__ = [
    "descriptor",
    "descriptor_pool",
    "runtime_version",
    "symbol_database",
    "builder",
    "grpc",
]
