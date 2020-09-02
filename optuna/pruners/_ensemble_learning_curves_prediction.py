import numpy as np
from scipy.stats import norm
from optuna.pruners import BasePruner
from optuna.trial._state import TrialState
from optuna import type_checking
import functools
import math

if type_checking.TYPE_CHECKING:
    from typing import KeysView  # NOQA
    from typing import List  # NOQA
    from optuna.study import Study  # NOQA
    from optuna.trial import FrozenTrial  # NOQA
    from optuna.storages import BaseStorage  # NOQA
    from optuna.structs import FrozenTrial  # NOQA


"""
[Reference]
Title  : Speeding up Hyper-parameter Optimization by Extrapolation of Learning Curves using Previous Builds
Authors: Akshay Chandrashekaran, Ian R. Lane
http://akshayc.com/publications/ecml2017_gelc_submitted.pdf
[Source Code]
https://github.com/akshayc11/Spearmint/blob/elc/spearmint/pylrpredictor/gradient_descent.py
https://github.com/akshayc11/Spearmint/blob/elc/spearmint/pylrpredictor/prevonlyterminationcriterion.py
"""


def recency_weights(num):
    """
    Parameters
    ----------
    num: int
        The number of epoch obtained up to the current iteration.
    Returns
    -------
    weights: np.ndarray (shape=(num,))
        The weights are generated based recency weighting to add to the residuals.
        This is done so that the loss from fresher epochs are given more weightage than loss from
        previous epochs
    """
    recency_weights = np.ones(num) * (10 ** (1. / num))
    return np.sqrt(recency_weights ** (np.arange(num)))


def _dict_to_increasing_list(intermediate_values, maximize):
    # type: (FrozenTrial, bool) -> np.ndarray
    vs, ks = map(np.asarray,
                 (intermediate_values.values(),
                  intermediate_values.keys()))

    order = np.argsort(ks)
    vs = vs[order]
    sz = vs.size()
    for i in range(sz - 1):
        vs[i + 1] = max(vs[i + 1], vs[i]) if maximize \
            else min(vs[i + 1], vs[i])

    return vs


def _is_first_in_interval_step(step, intermediate_steps, n_warmup_steps, interval_steps):
    # type: (int, KeysView[int], int, int) -> bool

    nearest_lower_pruning_step = (
        step - n_warmup_steps
    ) // interval_steps * interval_steps + n_warmup_steps
    assert nearest_lower_pruning_step >= 0

    # `intermediate_steps` may not be sorted so we must go through all elements.
    second_last_step = functools.reduce(
        lambda second_last_step, s: s if s > second_last_step and s != step else second_last_step,
        intermediate_steps,
        -1,
    )

    return second_last_step < nearest_lower_pruning_step


class EnsembleLearningCurvesPruner(BasePruner):
    """Pruner using Ensemble Learning Curves Prediction Algorithm.
    Parameters
    ----------
    max_epoch: int
        The number of epoch for training of task of interest.
    maximize: bool
        If True, the goal of the optimization is to maximize the objective function.
    yrange: tuple (shape=(2,))
        The range of the objective function.
        It is used for the termination criterion using standard deviation.
        If None is given, termination using standard deviation is not considered.
    n_startup_trials: int
        Early stopping will be implemented after evaluating this number of the objective function.
        The small number leads to less reliable early stopping.
        The large number leads to less efficiency.
    n_warmup_steps: int
        The evalution will be done up to at least this number of epoch.
    interval_steps: int
        The judgement of the cutting off will be implemented every this number step.
    n_estimators: int
        The number of estimator used in the ensemble model.
        The larger number leads to more computational time, but can lead to the stability.
    threshold: float
        If P(y_{curr, max_epoch} is better than y_{best, max_epoch}) < threshold,
        Current evaluation will be cut off.
    maximum_possible_std: float
        If standard deviation of y_{curr, max_epoch} is larger than
        (yrange[1] - yrange[0]) * maximum_possible_std, the cutting off will not be implemented.
        If np.inf is given, the cutting off by standard deviation will not be implemented.
    recency_weighting: bool
        If true, the predicted value fits fresher values more than older ones.
    monotonicity_condition: bool
        if true, the predicted accuracy of the estimator cannot be lesser than the best observed value so far.
        In other words, it prevents the wrong early stopping.
    seed: int
        Random Seed.
    alpha: float
        The coefficient of this gradient descent.
        In general, lower value leads to slower convergence, but reliable solutions
        Higher value leads to faster convergence, but unreliable solutions.
    n_iteration: int
        The number of iteration for the gradient descent.
    param1, param2, param3: float
        Hyperparameter of this method.
        param1: the regularization term.
        param2: the regularization term. This term variates depending on the number of epoch we refer to.
        param3: the parameter for monotonicity condition.
    n_prev: int
        The number of learning curves we refer to.
        This method picks up the specified number of learning curves from completed tasks
        ans uses them to construct estimators.
    """
    def __init__(self, max_epoch, maximize=False, yrange=None, n_startup_trials=5, n_warmup_steps=0, interval_steps=1, n_estimators=100,
                 threshold=0.05, maximum_possible_std=0.005, recency_weighting=False, monotonicity_condition=True, seed=None,
                 alpha=0.01, n_iteration=100, param1=1, param2=1, param3=5, n_prev=5):
        # type: (float, int, int, int) -> None

        if n_startup_trials < 0:
            raise ValueError(
                "The number of startup trials cannot be negative but got {}.".format(n_startup_trials)
            )
        if n_warmup_steps < 0:
            raise ValueError(
                "The number of warmup steps cannot be negative but got {}.".format(n_warmup_steps)
            )
        if n_estimators <= 0:
            raise ValueError(
                "The number of estimators must be positive but got {}.".format(n_estimators)
            )
        if n_iteration <= 0:
            raise ValueError(
                "The number of iteration for gradient descent must be positive but got {}.".format(n_iteration)
            )
        if interval_steps < 1:
            raise ValueError(
                "Pruning interval steps must be at least 1 but got {}.".format(interval_steps)
            )
        if alpha <= 0:
            raise ValueError(
                "Learning Rate of Gradient Descent must be positive but got {}.".format(alpha)
            )
        if n_prev <= 0:
            raise ValueError(
                "The number of previous learning curves to refer to must be positive but got {}.".format(n_prev)
            )
        if param1 < 0 or param2 < 0 or param3 < 0:
            raise ValueError(
                "The coefficients of gradient decesents cannot be negative but got ({}, {}, {}).".format(param1, param2, param3)
            )
        if threshold < 0 or threshold > 1:
            raise ValueError(
                "Pruning threshold must be between 0 and 1 but got {}.".format(threshold)
            )
        if yrange is not None:
            if not isinstance(yrange, tuple):
                raise TypeError(
                    "The type of yrange must be tuple or None but got {}.".format(type(tuple))
                )
            elif len(yrange) != 2:
                raise ValueError(
                    "The size of the range of Y must be 2 but got {}.".format(len(yrange))
                )
            elif yrange[0] > yrange[1]:
                raise ValueError(
                    "The upper bound of y must be larger or equal to the lower bound."
                )

        self._n_startup_trials = n_startup_trials
        self._n_warmup_steps = n_warmup_steps
        self._interval_steps = interval_steps
        self._n_estimators = n_estimators
        self._n_prev = n_prev
        self._max_epoch = max_epoch
        self._yrange = (-np.inf, np.inf) if yrange is None else yrange
        self._threshold = threshold
        self._recency_weighting = recency_weighting
        self._monotonicity_condition = monotonicity_condition
        self._maximum_possible_std = None if yrange is None or maximum_possible_std is None \
            else maximum_possible_std * (yrange[1] - yrange[0])
        self._maximize = maximize
        self._alpha = alpha
        self._n_iteration = n_iteration
        self.ps = [param1, param2, param3]
        self.rng = np.random.RandomState(seed)
        self.a, self.b, self.ids, self.ys_prev, self.y_curr = None, None, None, None, None

    def _gradient_descent(self, f_curr, f_prev):
        """
        Parameters
        ----------
        f_curr: np.ndarray (shape=(n_epoch,))
            The target of the current prediction.
            The observation is available until n_epoch epoch.
        f_prev: np.ndarray (shape=(n_epoch,))
            One of the previous builds for training.
            The observation is used until n_epoch epoch (We do not use after this epoch for training).
        Returns
        -------
        a, b: float
            The coefficient of affine transformation.
            This variable will be obtained as a result of gradient descent.
        residual: float
            The residual between the evaluation at each epoch of the current task
            and that of the affine-transformed previous task.
            *Note: This value is not apprently weighted in the original code.
        """

        assert(len(f_curr) == len(f_prev))

        if self._monotonicity_condition:
            f_curr_max = f_curr.max()
            f_prev_max = f_prev.max()

        n_epoch = len(f_curr)
        a = self.rng.rand()
        b = self.rng.rand() - 0.5
        weights = recency_weights(n_epoch)

        for i in range(self._n_iteration):
            res = f_curr - (a * f_prev + b)
            if self._recency_weighting:
                res = weights * res

            dL_da = - (res @ f_prev) / n_epoch - self.ps[0] * (1.0 - a) / np.exp(self.ps[1] * n_epoch)
            dL_db = - np.sum(res) / n_epoch

            if self._monotonicity_condition:
                monotonicity_val = np.exp(self.ps[2] * (f_curr_max - (a * f_prev_max + b)))
                dL_da -= self.ps[2] * f_prev_max * monotonicity_val
                dL_db -= self.ps[2] * monotonicity_val

            a -= self._alpha * dL_da
            b -= self._alpha * dL_db

        return a, b, np.linalg.norm(f_curr - (a * f_prev + b)) / n_epoch

    def _implement_gradient_descent(self, f_curr_org, f_prev_org, n_startpoints=100):
        # type: (np.ndarray, np.ndarray, int) -> tuple(np.ndarray, np.ndarray, np.ndarray)

        epoch = min(len(f_curr_org), len(f_prev_org))
        f_curr, f_prev = f_curr_org[:epoch], f_prev_org[:epoch]
        f_curr_max = f_curr_org.max()
        A, B, Ls = [], [], []

        for i in range(int(2 * n_startpoints)):
            a, b, L = self._gradient_descent(f_curr, f_prev)
            if not self._monotonicity_condition or a * f_prev[-1] + b >= f_curr_max:
                A.append(a), B.append(b), Ls.append(L)

            if len(A) == n_startpoints:
                break

        A, B, Ls = map(np.ndarray, (A, B, Ls))
        return A, B, Ls

    def fit(self):
        # type: () -> None

        if len(self.ys_prev) == 0:
            raise ValueError("No trials have been completed.")

        A, B, Ls, id = [], [], [], []
        for i, y_prev in enumerate(self.ys_prev):
            a, b, loss = self._implement_gradient_descent(self.y_curr, y_prev, n_startpoints=self._n_estimators)
            for ai, bi, li in zip(a, b, loss):
                A.append(ai), B.append(bi), Ls.append(li), id.append(i)

        A, B, Ls, id = map(np.asarray, (A, B, Ls, id))
        order = np.argsort(Ls)[:self._n_estimators]
        self.a, self.b, self.ids = A[order], B[order], id[order]

    def predict(self):
        # type: () -> tuple(float, float, bool)
        """
        Predict the mean and standard deviation of the extrapolation using
        previous model information
        P(y_{curr, max_epoch} | y_{curr, 1:m}; y_{prev, 1:m})
        """
        epoch = self._max_epoch

        if self.a is None or self.b is None or self.a.size == 0 or self.b.size == 0:
            return self._yrange[self._maximize], self._yrange[1] - self._yrange[0], False

        ys_pred = np.array([a * self.ys_prev[id][epoch - 1] + b for a, b, id in zip(self.a, self.b, self.ids)
                            if self._yrange[0] <= a * self.ys_prev[id][epoch - 1] + b <= self._yrange[1]])

        y_mean, y_std = np.mean(ys_pred), np.std(ys_pred)
        found = (self._yrange[0] <= y_mean <= self._yrange[1] and y_std >= 0.0)

        return y_mean, y_std, found

    def _posterior_probability_better_than(self, y_best, mean, std):
        # type: (float, float, float) -> float
        """
        Compute the following probability using the estimated mean and standard deviation.
        P(y_{curr, max_epoch} is better than y_{best, max_epoch})
        """
        p = norm.cdf(y_best, loc=mean, scale=std)

        return p if not self._maximize else 1.0 - p

    def prune(self, study, trial):
        # type: (BaseStorage, int, int, int) -> bool

        step = trial.last_step
        if step is None or step < self._n_warmup_steps:
            return False

        value = trial.intermediate_values[step]
        if math.isnan(value):
            return False

        all_trials = study.get_trials(deepcopy=False)
        self._maximize = study.direction
        self.ys_prev = [
            _dict_to_increasing_list(t.intermediate_values, self._maximize)
            for t in all_trials
            if t.state == TrialState.COMPLETE and not math.isnan(t.intermediate_values[step])]

        n_trials = len(self.ys_prev)
        if n_trials == 0 or n_trials < self._n_startup_trials:
            return False

        if len(self.ys_prev) > self._n_prev:
            index = self.rng.choice(np.arange(len(self.ys_prev)), self._n_prev, replace=False)
            self.ys_prev = self.ys_prev[index]

        self.y_curr = _dict_to_increasing_list(trial.intermediate_values, self._maximize)

        if not _is_first_in_interval_step(
            step, trial.intermediate_values.keys(), self._n_warmup_steps, self._interval_steps
        ):
            return False

        y_best, y_curr_best = self.ys_prev.max(), self.y_curr.max()
        if (self._maximize and y_curr_best >= y_best) or (not self._maximize and y_curr_best <= y_best):
            return False

        self.fit()
        mean, std, found = self.predict()
        if mean is None or std is None or not found:
            return False

        p_better = self._posterior_probability_better_than(y_best, mean, std)
        print('P(y is better than y_best) = {}\n'.format(p_better))

        if p_better < self._threshold:
            return self._maximum_possible_std is None or std <= self._maximum_possible_std

        return False
