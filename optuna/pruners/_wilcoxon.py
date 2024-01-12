from __future__ import annotations

import warnings

import numpy as np

import optuna
from optuna._experimental import experimental_class
from optuna._imports import _LazyImport
from optuna.pruners import BasePruner
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial


@experimental_class("3.6.0")
class WilcoxonPruner(BasePruner):
    """Pruner based on the Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.

    This pruner performs the Wilcoxon signed-rank test between the current trial and the current best trial,
    and stops whenever the pruner is sure up to a given p-value that the current trial is worse than the best one.

    This pruner is effective for objective functions (median, mean, etc.) that
    aggregates multiple evaluations.
    This includes the mean performance of n (e.g., 100)
    shuffled inputs, the mean performance of k-fold cross validation, etc.
    There can be "easy" or "hard" inputs (the pruner handles correspondence of
    the inputs between different trials),
    but it is recommended to shuffle the order of inputs once before the optimization.

    When you use this pruner, you must call `Trial.report(value, step)` function **for each `step = 1, 2, ..., N`** with
    the **past average values of evaluations** (`value=np.mean(evaluation_values[:step])`), regardless of the actual
    objective function. This interface is designed so that other pruners can be used interchangeably.

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

            # It is recommended to shuffle the input data once before optimization.
            np.random.shuffle(input_data)


            def objective(trial):
                s = 0.0
                for i in range(len(input_data)):
                    param = trial.suggest_uniform("param", -1, 1)
                    s += eval_func(param, input_data[i])

                    trial.report(s / (i + 1), i + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return s / len(input_data)


            study = optuna.study.create_study(
                pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1)
            )
            study.optimize(objective, n_trials=100)

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
        ss = _LazyImport("scipy.stats")

        def extract_step_values(t: FrozenTrial) -> np.ndarray | None:
            step = t.last_step
            if step is None:
                return np.empty((0,))

            all_steps = list(t.intermediate_values.keys())
            if all_steps != list(range(1, step + 1)):
                return None

            intermediate_average_values = np.array(
                [0] + [t.intermediate_values[step] for step in all_steps]
            )
            step_values = np.diff(intermediate_average_values * np.arange(step + 1))
            return step_values

        step = trial.last_step
        if step is None or step <= self._n_startup_steps:
            return False

        step_values = extract_step_values(trial)
        if step_values is None:
            raise ValueError(
                "WilcoxonPruner requires intermediate average values to be "
                "given at step [1, 2, ..., n], but got "
                f"{list(trial.intermediate_values.keys())} for trial {trial.number})."
            )

        if np.any(np.isnan(step_values)):
            warnings.warn(
                f"The intermediate values of the current trial (trial {trial.number}) "
                f"contain NaNs. WilcoxonPruner will not prune this trial."
            )
            return False

        try:
            best_trial = study.best_trial
        except ValueError:
            return False

        best_step_values = extract_step_values(best_trial)
        if best_step_values is None:
            warnings.warn(
                "WilcoxonPruner requires intermediate average values to be "
                "given at steps [1, 2, ..., n], but the best trial "
                f"(trial {best_trial.number}) has intermediate values at "
                f"steps {list(best_trial.intermediate_values.keys())}. "
                "WilcoxonPruner will not prune the current trial."
            )
            return False

        if len(best_step_values) < len(step_values):
            warnings.warn(
                f"The best trial (trial {best_trial.number}) has less "
                f"({len(best_step_values)}) steps than the current trial. Only the "
                f"first {len(best_step_values)} intermediate values will be compared."
            )
            step_values = step_values[: len(best_step_values)]
        elif len(best_step_values) > len(step_values):
            best_step_values = best_step_values[: len(step_values)]

        if np.any(np.isnan(best_step_values)):
            warnings.warn(
                f"The intermediate values of the best trial (trial {best_trial.number}) "
                f"contain NaNs. WilcoxonPruner will not prune the current trial."
            )
            return False

        diffs = step_values - best_step_values
        diffs[np.isnan(diffs)] = 0  # inf - inf or (-inf) - (-inf)
        if study.direction == StudyDirection.MAXIMIZE:
            diffs *= -1

        if len(diffs) == 0:
            return False

        p = ss.wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue
        return p < self._p_threshold
