import os
from typing import Dict

import optuna
from optuna._experimental import experimental
from optuna._imports import try_import


with try_import() as _imports:
    from tensorboard.plugins.hparams import api as hp
    import tensorflow as tf


@experimental("2.0.0")
class TensorBoardCallback(object):
    """Callback to track Optuna trials with TensorBoard.

    This callback adds relevant information that is tracked by Optuna to TensorBoard.

    See `the example <https://github.com/optuna/optuna/blob/master/
    examples/tensorboard_simple.py>`_.

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
            hparams[self._hp_params[param_name]] = param_value
        run_name = "trial-%d" % trial.number
        run_dir = os.path.join(self._dirname, run_name)
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams, trial_id=run_name)  # record the values used in this trial
            tf.summary.scalar(self._metric_name, trial_value, step=trial.number)

    def _add_distributions(
        self, distributions: Dict[str, optuna.distributions.BaseDistribution]
    ) -> None:
        for param_name, param_distribution in distributions.items():
            if isinstance(param_distribution, optuna.distributions.UniformDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name, hp.RealInterval(param_distribution.low, param_distribution.high)
                )
            elif isinstance(param_distribution, optuna.distributions.LogUniformDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name, hp.RealInterval(param_distribution.low, param_distribution.high)
                )
            elif isinstance(param_distribution, optuna.distributions.DiscreteUniformDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name, hp.Discrete(param_distribution.low, param_distribution.high)
                )
            elif isinstance(param_distribution, optuna.distributions.IntUniformDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name, hp.IntInterval(param_distribution.low, param_distribution.high)
                )
            elif isinstance(param_distribution, optuna.distributions.CategoricalDistribution):
                self._hp_params[param_name] = hp.HParam(
                    param_name, hp.Discrete(param_distribution.choices)
                )
            else:
                distribution_list = [
                    optuna.distributions.UniformDistribution.__name__,
                    optuna.distributions.LogUniformDistribution.__name__,
                    optuna.distributions.DiscreteUniformDistribution.__name__,
                    optuna.distributions.IntUniformDistribution.__name__,
                    optuna.distributions.CategoricalDistribution.__name__,
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
