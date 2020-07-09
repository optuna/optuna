from optuna.samplers._gp.model.base import BaseModel
from optuna.samplers._gp.model.gpy_exact import GPyExact


def model_selector(model: str) -> BaseModel:
    """Selector module for surrogate models."""

    if model == 'GPyExact':
        return GPyExact()
    else:
        raise ValueError("The surrogate model {} is not supported.".format(model))
