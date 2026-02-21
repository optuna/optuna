

try:
    from optuna.integration.cma import PyCamaSampler
except ModuleNotFoundError:
    raise ModuleNotFoundError(
            "\nCould not find `optuna-integration` for `cma`.\n"
            "Please run `pip install optuna-integration[cma]`.\n"
        )
        


__all__ = ["PyCamaSampler"]
