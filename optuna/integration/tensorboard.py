import optuna
from optuna import type_checking
import os

if type_checking.TYPE_CHECKING:
    from typing import Dict  # NOQA
    from typing import Optional  # NOQA

try:
    import tensorboard
    from tensorboard.plugins.hparams import api as hp
    import tensorflow as tf

    _available = True
except ImportError as e:
    _import_error = e
    _available = False
    hp = object
    tensorboard = object
    tensorflow = object


def _check_tensorboard_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "TensorBoard is not available. Please install TensorBoard to use this "
            "feature. It can be installed by executing `$ pip install "
            "tensorboard`. For further information, please refer to the installation guide "
            "of TensorBoard. (The actual import error is as follows: " + str(_import_error) + ")"
        )


class TensorBoardCallback(object):
    """Callback to track Optuna trials with TensorBoard.

    This callback adds relevant information that is tracked by Optuna to TensorBoard.
    Args:
        metric_name:
            Name of the metric. Since the metric itself is just a number,
            `metric_name` can be used to give it a name. So you know later
            if it was roc-auc or accuracy.
    """

    def __init__(self, dirname, param_distributions, metric_name):
        # type: (str, Dict[str, optuna.distributions.BaseDistribution], str) -> None

        _check_tensorboard_availability()

        self._dirname = dirname
        self._metric_name = metric_name
        self._hp_params = dict()  # type: Dict

        for param_name, param_distribution in param_distributions.items():

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

    def __call__(self, study, trial):
        # type: (optuna.study.Study, optuna.trial.FrozenTrial) -> None

        trial_value = trial.value if trial.value is not None else float("nan")

        hparams = dict()
        for param_name, param_value in trial.params.items():
            hparams[self._hp_params[param_name]] = param_value

        run_name = "trial-%d" % trial.number
        run_dir = os.path.join(self._dirname, run_name)

        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            tf.summary.scalar(self._metric_name, trial_value, step=1)
