from typing import Any
from typing import Dict
from typing import Optional

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import


with try_import() as _imports:
    import wandb


@experimental("2.9.0")
class WeightsAndBiasesCallback(object):
    """Callback to track Optuna trials with Weights & Biases.

    This callback enables tracking of Optuna study in
    Weights & Biases. The study is tracked as a single experiment
    run, where all suggested hyperparameters and optimized metric
    are logged and plotted as a function of optimizer steps.

    .. note::
        User needs to be logged in to Weights & Biases before
        using this callback in online mode. For more information, please
        refer to `wandb setup <https://docs.wandb.ai/quickstart#1-set-up-wandb>`_.

    .. note::
        Users who want to run multiple Optuna studies within the same process
        should call ``wandb.finish()`` between subsequent calls to
        ``study.optimize()``. Calling ``wandb.finish()`` is not necessary
        if you are running one Optuna study per process.

    .. note::
        To ensure correct trial order in Weights & Biases, this callback
        should only be used with ``study.optimize(n_jobs=1)``.

    .. note::
        This callback currently supports only single-objective optimization.

    Example:

        Add Weights & Biases callback to Optuna optimization.

        .. code::

            import optuna
            from optuna.integration.wandb import WeightsAndBiasesCallback


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            wandb_kwargs = {"project": "my-project"}
            wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)


            study = optuna.create_study(study_name="my_study")
            study.optimize(objective, n_trials=10, callbacks=[wandbc])

    Args:
        metric_name:
            Name of the optimized metric. Since the metric itself is just a number,
            ``metric_name`` can be used to give it a name. So you know later
            if it was roc-auc or accuracy.
        wandb_kwargs:
            Set of arguments passed when initializing Weights & Biases run.
            Please refer to `Weights & Biases API documentation
            <https://docs.wandb.ai/ref/python/init>`_ for more details.
    """

    def __init__(
        self, metric_name: str = "value", wandb_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:

        _imports.check()

        self._metric_name = metric_name
        self._wandb_kwargs = wandb_kwargs or {}

        self._initialize_run()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        direction = study.direction.name
        attributes = {"direction": direction}

        wandb.config.update(attributes)
        wandb.log({**trial.params, self._metric_name: trial.value}, step=trial.number)

    def _initialize_run(self) -> None:
        """Initializes Weights & Biases run."""

        wandb.init(**self._wandb_kwargs)
