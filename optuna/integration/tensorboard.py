import os
from typing import Dict

import optuna
from optuna._experimental import experimental_class
from optuna._imports import try_import
from optuna.logging import get_logger


with try_import() as _imports:
    from tensorboard.plugins.hparams import api as hp
    import tensorflow as tf

_logger = get_logger(__name__)


@experimental_class("2.0.0")
class TensorBoardCallback:
    """Callback to track Optuna trials with TensorBoard.

    This callback adds relevant information that is tracked by Optuna to TensorBoard.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    tensorboard/tensorboard_simple.py>`_.

    Args:
        dirname:
            Directory to store TensorBoard logs.
        metric_name:
            Name of the metric. Since the metric itself is just a number,
            `metric_name` can be used to give it a name. So you know later
            if it was roc-auc or accuracy.

    """

    def __init__(self, dirname: str, metric_name: str) -> None:
        _imports.check()
        self._dirname = dirname
        self._metric_name = metric_name
        self._hp_params: Dict[str, hp.HParam] = {}

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(self._hp_params) == 0:
            self._initialization(study)
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        trial_value = trial.value if trial.value is not None else float("nan")
        hparams = {}
        for param_name, param_value in trial.params.items():
            if param_name not in self._hp_params:
                self._add_distributions(trial.distributions)
            param = self._hp_params[param_name]
            if isinstance(param.domain, hp.Discrete):
                hparams[param] = param.domain.dtype(param_value)
            else:
                hparams[param] = param_value
        run_name = "trial-%d" % trial.number
        run_dir = os.path.join(self._dirname, run_name)
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams, trial_id=run_name)  # record the values used in this trial
            tf.summary.scalar(self._metric_name, trial_value, step=trial.number)

    def _add_distributions(
        self, distributions: Dict[str, optuna.distributions.BaseDistribution]
    ) -> None:
        supported_distributions = (
            optuna.distributions.CategoricalDistribution,
            optuna.distributions.FloatDistribution,
            optuna.distributions.IntDistribution,
        )

        for param_name, param_distribution in distributions.items():
            if isinstance(param_distribution, optuna.distributions.FloatDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name,
                    hp.RealInterval(float(param_distribution.low), float(param_distribution.high)),
                )
            elif isinstance(param_distribution, optuna.distributions.IntDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name,
                    hp.IntInterval(param_distribution.low, param_distribution.high),
                )
            elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                choices = param_distribution.choices
                dtype = type(choices[0])
                if any(not isinstance(choice, dtype) for choice in choices):
                    _logger.warning(
                        "Choices contains mixed types, which is not supported by TensorBoard. "
                        "Converting all choices to strings."
                    )
                    choices = tuple(map(str, choices))
                elif dtype not in (int, float, bool, str):
                    _logger.warning(
                        f"Choices are of type {dtype}, which is not supported by TensorBoard. "
                        "Converting all choices to strings."
                    )
                    choices = tuple(map(str, choices))

                self._hp_params[param_name] = hp.HParam(
                    param_name,
                    hp.Discrete(choices),
                )
            else:
                distribution_list = [
                    distribution.__name__ for distribution in supported_distributions
                ]
                raise NotImplementedError(
                    "The distribution {} is not implemented. "
                    "The parameter distribution should be one of the {}".format(
                        param_distribution, distribution_list
                    )
                )

    def _initialization(self, study: optuna.Study) -> None:
        completed_trials = [
            trial
            for trial in study.get_trials(deepcopy=False)
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        for trial in completed_trials:
            self._add_distributions(trial.distributions)
