from optuna._imports import try_import


with try_import() as _imports:
    import grpc

    from optuna.storages.grpc._auto_generated import api_pb2
    from optuna.storages.grpc._auto_generated import api_pb2_grpc


__all__ = [
    "grpc",
    "api_pb2",
    "api_pb2_grpc",
]
