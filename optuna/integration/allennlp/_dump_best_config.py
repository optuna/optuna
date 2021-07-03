import json

import optuna
from optuna._imports import try_import
from optuna.integration.allennlp._environment import _environment_variables


with try_import() as _imports:
    import _jsonnet


def dump_best_config(input_config_file: str, output_config_file: str, study: optuna.Study) -> None:
    """Save JSON config file with environment variables and best performing hyperparameters.

    Args:
        input_config_file:
            Input Jsonnet config file used with
            :class:`~optuna.integration.AllenNLPExecutor`.
        output_config_file:
            Output JSON config file.
        study:
            Instance of :class:`~optuna.study.Study`.
            Note that :func:`~optuna.study.Study.optimize` must have been called.

    """
    _imports.check()

    # Get environment variables.
    ext_vars = _environment_variables()

    # Get the best hyperparameters.
    best_params = study.best_params
    for key, value in best_params.items():
        best_params[key] = str(value)

    # If keys both appear in environment variables and best_params,
    # values in environment variables are overwritten, which means best_params is prioritized.
    ext_vars.update(best_params)

    best_config = json.loads(_jsonnet.evaluate_file(input_config_file, ext_vars=ext_vars))

    # `optuna_pruner` only works with Optuna.
    # It removes when dumping configuration since
    # the result of `dump_best_config` can be passed to
    # `allennlp train`.
    if "callbacks" in best_config["trainer"]:
        new_callbacks = []
        callbacks = best_config["trainer"]["callbacks"]
        for callback in callbacks:
            if callback["type"] == "optuna_pruner":
                continue
            new_callbacks.append(callback)

        if len(new_callbacks) == 0:
            best_config["trainer"].pop("callbacks")
        else:
            best_config["trainer"]["callbacks"] = new_callbacks

    with open(output_config_file, "w") as f:
        json.dump(best_config, f, indent=4)
