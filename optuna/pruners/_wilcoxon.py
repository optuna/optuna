from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial


if TYPE_CHECKING:
    import scipy.stats as ss
else:
    from optuna._imports import _LazyImport

    ss = _LazyImport("scipy.stats")


@experimental_class("3.6.0")
class WilcoxonPruner(BasePruner):
    """Pruner based on the `Wilcoxon signed-rank test <https://en.wikipedia.org/w/index.php?title=Wilcoxon_signed-rank_test&oldid=1195011212>`_.

    This pruner performs the Wilcoxon signed-rank test between the current trial and the current best trial,
    and stops whenever the pruner is sure up to a given p-value that the current trial is worse than the best one.

    This pruner is effective for objective functions (median, mean, etc.) that
    aggregates multiple evaluations.
    This includes the mean performance of n (e.g., 100)
    shuffled inputs, the mean performance of k-fold cross validation, etc.
    There can be "easy" or "hard" inputs (the pruner handles correspondence of
    the inputs between different trials).
    In each trial, it is recommended to shuffle the order in which data is processed.

    When you use this pruner, you must call `Trial.report(value, step)` function for each step (e.g., input id) with
    the evaluated value. This is different from other pruners in that the reported value need not converge
    to the real value. (To use pruners such as `SuccessiveHalvingPruner` in the same setting, you must provide e.g.,
    the historical average of the evaluated values.)

    .. seealso::
        Please refer to :meth:`~optuna.trial.Trial.report`.

    Example:

        .. testcode::

            import optuna
            import numpy as np


            # For demonstrative purposes, we will use a toy evaluation function.
            # We will minimize the mean value of `eval_func` over the input dataset.
            def eval_func(param, input_):
                return (param - input_) ** 2


            input_data = np.linspace(-1, 1, 100)
            rng = np.random.default_rng()


            def objective(trial):
                # In each trial, it is recommended to shuffle the order in which data is processed.
                ordering = rng.permutation(range(len(input_data)))
                s = []
                for i in ordering:
                    param = trial.suggest_float("param", -1, 1)
                    loss = eval_func(param, input_data[i])
                    trial.report(loss, i)
                    s.append(loss)
                    if trial.should_prune():
                        return sum(s) / len(s)  # An advanced workaround (see the note below).
                        # raise optuna.TrialPruned()

                return sum(s) / len(s)


            study = optuna.study.create_study(
                pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1)
            )
            study.optimize(objective, n_trials=100)


    .. note::
        This pruner cannot handle ``infinity`` or ``nan`` values.
        Trials containing those values are never pruned.

    .. note::
        As an advanced workaround, if `trial.should_prune()` returns `True`,
        you can return an estimation of the final value (e.g., the average of all evaluated values)
        instead of `raise optuna.TrialPruned()`.
        Some algorithms including `TPESampler` internally split trials into below (good) and above (bad),
        and pruned trial will always be classified as above.
        However, there are some trials that are slightly worse than the best trial and will be pruned,
        but they should be classified as below (e.g., top 10%).
        This workaround provides beneficial information about such trials to these algorithms.

    Args:
        p_threshold:
            The p-value threshold for pruning. This value should be between 0 and 1.
            A trial will be pruned whenever the pruner is sure up to the given p-value
            that the current trial is worse than the best trial.
            The larger this value is, the more aggressive pruning will be performed.
            Defaults to 0.1.

            .. note::
                Contrary to the usual statistical wisdom, this pruner repeatedly
                performs statistical tests between the current trial and the
                current best trial with increasing samples.
                Please expect around ~2x probability of falsely pruning
                a good trial, compared to the usual false positive rate of
                performing the statistical test only once.

        n_startup_steps:
            The number of steps before which no trials are pruned.
            Pruning starts only after you have `n_startup_steps` steps of
            available observations for comparison between the current trial
            and the best trial.
            Defaults to 0 (pruning kicks in from the very first step).
    """  # NOQA: E501

    def __init__(
        self,
        *,
        p_threshold: float = 0.1,
        n_startup_steps: int = 0,
    ) -> None:
        if n_startup_steps < 0:
            raise ValueError(f"n_startup_steps must be nonnegative but got {n_startup_steps}.")
        if not 0.0 <= p_threshold <= 1.0:
            raise ValueError(f"p_threshold must be between 0 and 1 but got {p_threshold}.")

        self._n_startup_steps = n_startup_steps
        self._p_threshold = p_threshold

    def prune(self, study: "optuna.study.Study", trial: FrozenTrial) -> bool:
        if len(trial.intermediate_values) == 0:
            return False

        steps, step_values = np.array(list(trial.intermediate_values.items())).T

        if np.any(~np.isfinite(step_values)):
            warnings.warn(
                f"The intermediate values of the current trial (trial {trial.number}) "
                f"contain infinity/NaNs. WilcoxonPruner will not prune this trial."
            )
            return False

        try:
            best_trial = study.best_trial
        except ValueError:
            return False

        best_steps, best_step_values = np.array(list(best_trial.intermediate_values.items())).T

        if np.any(~np.isfinite(best_step_values)):
            warnings.warn(
                f"The intermediate values of the best trial (trial {best_trial.number}) "
                f"contain infinity/NaNs. WilcoxonPruner will not prune the current trial."
            )
            return False

        _, idx1, idx2 = np.intersect1d(steps, best_steps, return_indices=True)

        if len(idx1) < len(step_values):
            # This if-statement is never satisfied if following "average_is_best" safety works,
            # because the safety ensures that the best trial always has the all steps.
            warnings.warn(
                "WilcoxonPruner finds steps existing in the current trial "
                "but does not exist in the best trial. "
                "Those values are ignored."
            )

        diff_values = step_values[idx1] - best_step_values[idx2]

        if len(diff_values) < self._n_startup_steps:
            return False

        if study.direction == StudyDirection.MAXIMIZE:
            alt = "less"
            average_is_best = best_trial.value <= sum(step_values) / len(step_values)
        else:
            alt = "greater"
            average_is_best = best_trial.value >= sum(step_values) / len(step_values)

        # We use zsplit to avoid the problem when all values are zero.
        p = ss.wilcoxon(diff_values, alternative=alt, zero_method="zsplit").pvalue

        if p < self._p_threshold and average_is_best:
            # ss.wilcoxon found the current trial is probably worse than the best trial,
            # but the value of the best trial was not better than
            # the average of the current trial's intermediate values.
            # For safety, WilcoxonPruner concludes not to prune it for now.
            return False
        return p < self._p_threshold
