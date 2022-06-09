import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Union

import optuna
from optuna._experimental import experimental_class
from optuna._experimental import experimental_func
from optuna._imports import try_import
from optuna.study.study import ObjectiveFuncType


with try_import() as _imports:
    import wandb


@experimental_class("2.9.0")
class WeightsAndBiasesCallback:
    """Callback to track Optuna trials with Weights & Biases.

    This callback enables tracking of Optuna study in
    Weights & Biases. The study is tracked as a single experiment
    run, where all suggested hyperparameters and optimized metrics
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


    Example:

        Add Weights & Biases callback to Optuna optimization.

        .. code::

            import optuna
            from optuna.integration.wandb import WeightsAndBiasesCallback


            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()

            wandb_kwargs = {"project": "my-project"}
            wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs)

            study.optimize(objective, n_trials=10, callbacks=[wandbc])



        Weights & Biases logging in multirun mode.

        .. code::

            import optuna
            from optuna.integration.wandb import WeightsAndBiasesCallback

            wandb_kwargs = {"project": "my-project"}
            wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)


            @wandbc.track_in_wandb()
            def objective(trial):
                x = trial.suggest_float("x", -10, 10)
                return (x - 2) ** 2


            study = optuna.create_study()
            study.optimize(objective, n_trials=10, callbacks=[wandbc])


    Args:
        metric_name:
            Name assigned to optimized metric. In case of multi-objective optimization,
            list of names can be passed. Those names will be assigned
            to metrics in the order returned by objective function.
            If single name is provided, or this argument is left to default value,
            it will be broadcasted to each objective with a number suffix in order
            returned by objective function e.g. two objectives and default metric name
            will be logged as ``value_0`` and ``value_1``. The number of metrics must be
            the same as the number of values objective function returns.
        wandb_kwargs:
            Set of arguments passed when initializing Weights & Biases run.
            Please refer to `Weights & Biases API documentation
            <https://docs.wandb.ai/ref/python/init>`_ for more details.
        as_multirun:
            Creates new runs for each trial. Useful for generating W&B Sweeps like
            panels (for ex., parameter importance, parallel coordinates, etc).

    """

    def __init__(
        self,
        metric_name: Union[str, Sequence[str]] = "value",
        wandb_kwargs: Optional[Dict[str, Any]] = None,
        as_multirun: bool = False,
    ) -> None:

        _imports.check()

        if not isinstance(metric_name, Sequence):
            raise TypeError(
                "Expected metric_name to be string or sequence of strings, got {}.".format(
                    type(metric_name)
                )
            )

        self._metric_name = metric_name
        self._wandb_kwargs = wandb_kwargs or {}
        self._as_multirun = as_multirun

        if not self._as_multirun:
            self._initialize_run()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:

        if isinstance(self._metric_name, str):
            if len(trial.values) > 1:
                # Broadcast default name for multi-objective optimization.
                names = ["{}_{}".format(self._metric_name, i) for i in range(len(trial.values))]

            else:
                names = [self._metric_name]

        else:
            if len(self._metric_name) != len(trial.values):
                raise ValueError(
                    "Running multi-objective optimization "
                    "with {} objective values, but {} names specified. "
                    "Match objective values and names, or use default broadcasting.".format(
                        len(trial.values), len(self._metric_name)
                    )
                )

            else:
                names = [*self._metric_name]

        metrics = {name: value for name, value in zip(names, trial.values)}

        if self._as_multirun:
            metrics["trial_number"] = trial.number

        attributes = {"direction": [d.name for d in study.directions]}

        step = trial.number if wandb.run else None
        run = wandb.run

        # Might create extra runs if a user logs in wandb but doesn't use the decorator.

        if not run:
            run = self._initialize_run()
            run.name = f"trial/{trial.number}/{run.name}"

        run.log({**trial.params, **metrics}, step=step)

        if self._as_multirun:
            run.config.update({**attributes, **trial.params})
            run.tags = tuple(self._wandb_kwargs.get("tags", ())) + (study.study_name,)
            run.finish()
        else:
            run.config.update(attributes)

    @experimental_func("3.0.0")
    def track_in_wandb(self) -> Callable:
        """Decorator for using W&B for logging inside the objective function.

        The run is initialized with the same ``wandb_kwargs`` that are passed to the callback.
        All the metrics from inside the objective function will be logged into the same run
        which stores the parameters for a given trial.

        Example:

            Add additional logging to Weights & Biases.

            .. code::

                import optuna
                from optuna.integration.wandb import WeightsAndBiasesCallback
                import wandb

                wandb_kwargs = {"project": "my-project"}
                wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)


                @wandbc.track_in_wandb()
                def objective(trial):
                    x = trial.suggest_float("x", -10, 10)
                    wandb.log({"power": 2, "base of metric": x - 2})

                    return (x - 2) ** 2


                study = optuna.create_study()
                study.optimize(objective, n_trials=10, callbacks=[wandbc])


        Returns:
            ObjectiveFuncType: Objective function with W&B tracking enabled.
        """

        def decorator(func: ObjectiveFuncType) -> ObjectiveFuncType:
            @functools.wraps(func)
            def wrapper(trial: optuna.trial.Trial) -> Union[float, Sequence[float]]:

                run = wandb.run  # Uses global run when `as_multirun` is set to False.
                if not run:
                    run = self._initialize_run()
                    run.name = f"trial/{trial.number}/{run.name}"

                return func(trial)

            return wrapper

        return decorator

    def _initialize_run(self) -> "wandb.sdk.wandb_run.Run":
        """Initializes Weights & Biases run."""
        run = wandb.init(**self._wandb_kwargs)
        if not isinstance(run, wandb.sdk.wandb_run.Run):
            raise RuntimeError(
                "Cannot create a Run. "
                "Expected wandb.sdk.wandb_run.Run as a return. "
                f"Got: {type(run)}."
            )
        return run
