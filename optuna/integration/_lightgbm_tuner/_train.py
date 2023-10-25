from __future__ import annotations

from collections.abc import Callable
from typing import Any

from optuna._imports import try_import
from optuna.integration._lightgbm_tuner.optimize import _imports
from optuna.integration._lightgbm_tuner.optimize import LightGBMTuner
from optuna.study import Study
from optuna.trial import FrozenTrial


with try_import():
    import lightgbm as lgb


def train(
    params: dict[str, Any],
    train_set: "lgb.Dataset",
    num_boost_round: int = 1000,
    valid_sets: list["lgb.Dataset"] | tuple["lgb.Dataset", ...] | "lgb.Dataset" | None = None,
    valid_names: Any | None = None,
    feval: Callable[..., Any] | None = None,
    feature_name: str = "auto",
    categorical_feature: str = "auto",
    keep_training_booster: bool = False,
    callbacks: list[Callable[..., Any]] | None = None,
    time_budget: int | None = None,
    sample_size: int | None = None,
    study: Study | None = None,
    optuna_callbacks: list[Callable[[Study, FrozenTrial], None]] | None = None,
    model_dir: str | None = None,
    verbosity: int | None = None,
    show_progress_bar: bool = True,
    *,
    optuna_seed: int | None = None,
) -> "lgb.Booster":
    """Wrapper of LightGBM Training API to tune hyperparameters.

    It optimizes the following hyperparameters in a stepwise manner:
    ``lambda_l1``, ``lambda_l2``, ``num_leaves``, ``feature_fraction``, ``bagging_fraction``,
    ``bagging_freq`` and ``min_child_samples``.
    It is a drop-in replacement for `lightgbm.train()`_. See
    `a simple example of LightGBM Tuner <https://github.com/optuna/optuna-examples/tree/main/
    lightgbm/lightgbm_tuner_simple.py>`_ which optimizes the validation log loss of cancer
    detection.

    :func:`~optuna.integration.lightgbm.train` is a wrapper function of
    :class:`~optuna.integration.lightgbm.LightGBMTuner`. To use feature in Optuna such as
    suspended/resumed optimization and/or parallelization, refer to
    :class:`~optuna.integration.lightgbm.LightGBMTuner` instead of this function.

    .. note::
        Arguments and keyword arguments for `lightgbm.train()`_ can be passed.
        For ``params``, please check `the official documentation for LightGBM
        <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_.

    Args:
        time_budget:
            A time budget for parameter tuning in seconds.

        study:
            A :class:`~optuna.study.Study` instance to store optimization results. The
            :class:`~optuna.trial.Trial` instances in it has the following user attributes:
            ``elapsed_secs`` is the elapsed time since the optimization starts.
            ``average_iteration_time`` is the average time of iteration to train the booster
            model in the trial. ``lgbm_params`` is a JSON-serialized dictionary of LightGBM
            parameters used in the trial.

        optuna_callbacks:
            List of Optuna callback functions that are invoked at the end of each trial.
            Each function must accept two parameters with the following types in this order:
            :class:`~optuna.study.Study` and :class:`~optuna.trial.FrozenTrial`.
            Please note that this is not a ``callbacks`` argument of `lightgbm.train()`_ .

        model_dir:
            A directory to save boosters. By default, it is set to :obj:`None` and no boosters are
            saved. Please set shared directory (e.g., directories on NFS) if you want to access
            :meth:`~optuna.integration.lightgbm.LightGBMTuner.get_best_booster` in distributed
            environments. Otherwise, it may raise :obj:`ValueError`. If the directory does not
            exist, it will be created. The filenames of the boosters will be
            ``{model_dir}/{trial_number}.pkl`` (e.g., ``./boosters/0.pkl``).

        verbosity:
            A verbosity level to change Optuna's logging level. The level is aligned to
            `LightGBM's verbosity`_ .

            .. warning::
                Deprecated in v2.0.0. ``verbosity`` argument will be removed in the future.
                The removal of this feature is currently scheduled for v4.0.0,
                but this schedule is subject to change.

                Please use :func:`~optuna.logging.set_verbosity` instead.

        show_progress_bar:
            Flag to show progress bars or not. To disable progress bar, set this :obj:`False`.

            .. note::
                Progress bars will be fragmented by logging messages of LightGBM and Optuna.
                Please suppress such messages to show the progress bars properly.

        optuna_seed:
            ``seed`` of :class:`~optuna.samplers.TPESampler` for random number generator
            that affects sampling for ``num_leaves``, ``bagging_fraction``, ``bagging_freq``,
            ``lambda_l1``, and ``lambda_l2``.

            .. note::
                The `deterministic`_ parameter of LightGBM makes training reproducible.
                Please enable it when you use this argument.

    .. _lightgbm.train(): https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html
    .. _LightGBM's verbosity: https://lightgbm.readthedocs.io/en/latest/Parameters.html#verbosity
    .. _deterministic: https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic
    """
    _imports.check()

    auto_booster = LightGBMTuner(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        feature_name=feature_name,
        categorical_feature=categorical_feature,
        keep_training_booster=keep_training_booster,
        callbacks=callbacks,
        time_budget=time_budget,
        sample_size=sample_size,
        study=study,
        optuna_callbacks=optuna_callbacks,
        model_dir=model_dir,
        verbosity=verbosity,
        show_progress_bar=show_progress_bar,
        optuna_seed=optuna_seed,
    )
    auto_booster.run()
    return auto_booster.get_best_booster()
