

try:
    from optuna.integration.cma import PyCamaSampler
except ModuleNotFoundError:
    raise ModuleNotFoundError((
        f"\nCould not find `optuna-integration` for `cma`.\n"
        f"Please run `pip install optuna-integration[cma]`.\n"
    ))


__all__ = ["PyCamaSampler"]
